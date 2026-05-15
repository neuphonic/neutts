from functools import partial

from datasets import load_dataset, load_from_disk
from fire import Fire
from phonemizer.backend import EspeakBackend


def build_phonemizer(language):
    return EspeakBackend(
        language=language,
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch="ignore",
        language_switch="remove-flags",
    )


def normalize_phones(phones):
    return " ".join(phones.split())


def phonemize_sample(sample, g2p, text_column, phones_column):
    text = sample[text_column]
    phones = g2p.phonemize([text])
    sample[phones_column] = normalize_phones(phones[0]) if phones and phones[0] else ""
    return sample


def main(
    dataset: str,
    output_path: str = None,
    split: str = "train",
    language: str = "en-us",
    text_column: str = "text",
    phones_column: str = "phones",
    data_files: str = None,
    push_to_hub: str = None,
    load_from_disk_path: bool = False,
    num_proc: int = None,
):
    if output_path is None and push_to_hub is None:
        raise ValueError("Provide `output_path` to save locally or `push_to_hub` to upload.")

    if load_from_disk_path:
        phonemized_dataset = load_from_disk(dataset)
        if split and hasattr(phonemized_dataset, "keys"):
            phonemized_dataset = phonemized_dataset[split]
    else:
        phonemized_dataset = load_dataset(dataset, split=split, data_files=data_files)

    if text_column not in phonemized_dataset.column_names:
        raise ValueError(f"Dataset must contain a `{text_column}` column.")

    g2p = build_phonemizer(language)
    phonemized_dataset = phonemized_dataset.map(
        partial(
            phonemize_sample,
            g2p=g2p,
            text_column=text_column,
            phones_column=phones_column,
        ),
        num_proc=num_proc,
        desc=f"Phonemizing `{text_column}` to `{phones_column}`",
    )

    if output_path:
        phonemized_dataset.save_to_disk(output_path)

    if push_to_hub:
        phonemized_dataset.push_to_hub(push_to_hub)


if __name__ == "__main__":
    Fire(main)
