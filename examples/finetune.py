import warnings

import re
import os
import torch
from phonemizer.backend import EspeakBackend

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
from datasets import load_dataset, load_from_disk

warnings.filterwarnings("ignore")


ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")
DEFAULT_LANGUAGE = "en-us"


def build_phonemizer(language):
    return EspeakBackend(
        language=language,
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags",
    )


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


def normalize_phones(phones):
    if isinstance(phones, list):
        phones = " ".join(str(phone) for phone in phones)

    return " ".join(str(phones).split())


def get_phones(sample, g2p):
    if "phones" in sample and sample["phones"]:
        return normalize_phones(sample["phones"])

    if g2p is None:
        LOGGER.warning(f"⚠️ Missing phones for sample: {sample.get('__key__', '<unknown>')}")
        return None

    text = sample.get("text")
    if not text:
        LOGGER.warning(f"⚠️ Missing text for sample: {sample.get('__key__', '<unknown>')}")
        return None

    phones = g2p.phonemize([text])
    if not phones or not phones[0]:
        LOGGER.warning(
            f"⚠️ Empty phonemization output for sample: {sample.get('__key__', '<unknown>')} "
            f"text={text}"
        )
        return None

    return normalize_phones(phones[0])


def preprocess_sample(sample, tokenizer, max_len, g2p=None):

    # get special tokens
    speech_gen_start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    ignore_index = -100  # this is from LLaMA

    # unpack sample
    vq_codes = sample["codes"]

    phones = get_phones(sample, g2p)
    if not phones:
        return None

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

    language = config.get("language", DEFAULT_LANGUAGE)
    if "language" not in config:
        LOGGER.warning(
            f"No `language` found in config; defaulting to `{DEFAULT_LANGUAGE}`. "
            "Add `language` to your finetune config to control phonemization."
        )

    print(f"Loading checkpoint from {restore_from}")
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    model = AutoModelForCausalLM.from_pretrained(restore_from, torch_dtype="auto")

    dataset_name = config.get("dataset_name", "neuphonic/emilia-yodas-english-neucodec")
    dataset_split = config.get("dataset_split", "train[:2000]")
    dataset_from_disk = config.get("dataset_from_disk", False)

    if dataset_from_disk:
        emilia_dataset = load_from_disk(dataset_name)
        if dataset_split and hasattr(emilia_dataset, "keys"):
            emilia_dataset = emilia_dataset[dataset_split]
    else:
        emilia_dataset = load_dataset(dataset_name, split=dataset_split)

    g2p = None
    if "phones" not in emilia_dataset.column_names:
        LOGGER.warning(
            "Dataset has no `phones` column; phonemizing `text` inline is deprecated. "
            "Run `python examples/phonemize_dataset.py ...` once to create a pre-phonemized "
            "dataset before finetuning."
        )
        g2p = build_phonemizer(language)

    partial_preprocess = partial(
        preprocess_sample,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p,
    )

    if "text" in emilia_dataset.column_names:
        emilia_dataset = emilia_dataset.filter(data_filter)
    else:
        LOGGER.warning("Dataset has no `text` column; skipping text-based filtering.")

    emilia_dataset = emilia_dataset.map(
        partial_preprocess, remove_columns=emilia_dataset.column_names
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
