# Model finetuning

NeuTTS-Air follows [Llasa](https://github.com/zhenye234/LLaSA_training) in its training and inference setup. In order to finetune a model, you can use the `transformers` library from Hugging Face. We have an [example script](/examples/finetune.py) for finetuning using the [Emilia-YODAS dataset](https://huggingface.co/datasets/neuphonic/emilia-yodas-english-neucodec) that is encoded with [NeuCodec](https://huggingface.co/neuphonic/neucodec).

> [!NOTE]
> We have an on-going discussion about finetuning [here](https://github.com/neuphonic/neutts-air/issues/7) where some users have reported success with finetuning using the example script.

# Finetuning on your own dataset

You can prepare your own dataset by following these steps:

1. Encode your audio files using the [NeuCodec](https://huggingface.co/neuphonic/neucodec) model into a format similar to the [Emilia-YODAS dataset](https://huggingface.co/datasets/neuphonic/emilia-yodas-english-neucodec).
2. Setup your configuration file similar to the [example config](/examples/finetune_config.yaml).
3. Phonemize your text once before training so your dataset has a `phones` column. For a Hugging Face dataset, run:

    ```bash
    python examples/phonemize_dataset.py DATASET_NAME ./phonemized_dataset --language en-us
    ```

    Replace `DATASET_NAME`, the output path, and `--language` with values for your dataset. Use `python examples/phonemize_dataset.py parquet ./phonemized_dataset --data_files path/to/data.parquet --language en-us` for local parquet files. See [the phonemizer documentation](https://bootphon.github.io/phonemizer/api_reference.html#phonemizer.backend.espeak.espeak.EspeakBackend) for supported eSpeak language codes.
4. Run the finetuning script with your dataset and configuration file. To do this, navigate to the base directory of your cloned repo in the terminal and run:

    ```bash
    python examples/finetune.py examples/finetune_config.yaml
    ```

    replacing the argument with the path to your own config file if needed.

The finetuning script reads `phones` directly when the column is present. Datasets without `phones` still work through inline phonemization for compatibility, but preprocessing first is recommended.

# Finetuning config

An example finetuning config lives in `examples/finetune_config.yaml`.

- Set `dataset_name`, `dataset_split`, and `dataset_from_disk` to point at your training dataset. For the local output above, use `dataset_name: "./phonemized_dataset"` and `dataset_from_disk: true`.
- Set `language` to the eSpeak language code that matches your dataset, for example `en-us`, `de`, `fr-fr`, or `es`.
- In the past we've found a learning rate of `1e-5` to `4e-5` to have worked well for finetuning depending on the size of the dataset.
- We generally find that you do not need many steps for finetuning. For example, for a dataset of 10 hours, 1000 to 2000 steps is often sufficient.
- A warmup ratio as well as different learning rate schedulers can be experimented with to see what works best for your dataset.

# Training from scratch or using additional labels

The NeuTTS Air model is based on the [Qwen2.5 0.5B model](https://huggingface.co/Qwen/Qwen2.5-0.5B). To use this instead of the trained NeuTTS Air model, change the `restore_from` parameter in your config file to `"Qwen/Qwen2.5-0.5B"`.

Using Qwen means you would need to add the speech token tags to the model vocabulary. With either Qwen or NeuTTS you can also add additional custom tags. Both of these steps can be done as such in the script after loading the model:

```python
codec_special_tokens = [
    # speech token tags to add if using Qwen
    "<|TEXT_REPLACE|>",
    "<|TEXT_PROMPT_START|>",
    "<|TEXT_PROMPT_END|>",
    "<|SPEECH_REPLACE|>",
    "<|SPEECH_GENERATION_START|>",
    "<|SPEECH_GENERATION_END|>",
    # optional additional tags that you can add to enable features if you have the labels in your dataset
    "<|EN|>",
    "<|ZH|>",
    "<|LAUGHING|>",
    "<|WHISPERING|>",
]
codec_tokens = [f"<|speech_{idx}|>" for idx in range(config.codebook_size)]

new_tokens = codec_special_tokens + codec_tokens
n_added_tokens = tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
model.vocab_size = len(tokenizer)
```

You can then modify the input to the model to include these additional labels. For example, if you have speaker IDs or emotion labels, you can concatenate them with the phoneme tokens before passing them to the model.
