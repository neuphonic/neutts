import warnings

import re
import os
import torch
import phonemizer

from fire import Fire
from omegaconf import OmegaConf
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from loguru import logger as LOGGER
from datasets import load_dataset

warnings.filterwarnings("ignore")


ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")


def data_filter(sample):
    text = sample["text"]

    if len(text) == 0:
        return False

    if re.search(r"\d", text):
        return False

    if re.search(ACRONYM, text) or re.search(ACRONYM_NO_PERIOD, text):
        return False

    if text[-1] not in ".,?!":
        return False

    if "£" in text or "$" in text:
        return False

    return True


def preprocess_sample(sample, tokenizer, max_len, g2p):

    # get special tokens
    speech_gen_start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    ignore_index = -100  # this is from LLaMA

    # unpack sample
    vq_codes = sample["codes"]
    text = sample["text"]

    # phonemize
    phones = g2p.phonemize([text])

    # SAFE CHECK
    if not phones or not phones[0]:
        LOGGER.warning(f"⚠️ Empty phonemization output for sample: {sample['__key__']} text={text}")
        return None

    phones = phones[0].split()
    phones = " ".join(phones)

    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])

    # get chat format
    chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""  # noqa
    ids = tokenizer.encode(chat)

    # pad to make seq len
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    # convert to tensor
    input_ids = torch.tensor(ids, dtype=torch.long)

    labels = torch.full_like(input_ids, ignore_index)
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]

    # create attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # return in hf format
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main(config_fpath: str):

    # load config
    print(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    LOGGER.info(f"Logging to: {checkpoints_dir}")

    restore_from = config.restore_from

    print(f"Loading checkpoint from {restore_from}")
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    model = AutoModelForCausalLM.from_pretrained(restore_from, torch_dtype="auto")

    g2p = phonemizer.backend.EspeakBackend(
        language="en-us",
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags",
    )
    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
    )

    emilia_dataset = load_dataset(
        "neuphonic/emilia-yodas-english-neucodec",
        split="train[:2000]",
    )
    emilia_dataset = emilia_dataset.filter(data_filter).map(
        partial_preprocess, remove_columns=["text", "codes"]
    )

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        bf16=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        torch_compile=True,
        dataloader_num_workers=64,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=emilia_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(checkpoints_dir)


if __name__ == "__main__":
    Fire(main)
