# Model finetuning

NeuTTS-Air follows [Llasa](https://github.com/zhenye234/LLaSA_training) in its training and inference setup. In order to finetune a model, you can use the `transformers` library from Hugging Face. We have an [example script](/examples/finetune.py) for finetuning using the [Emilia-YODAS dataset](https://huggingface.co/datasets/neuphonic/emilia-yodas-english-neucodec) that is encoded with [NeuCodec](https://huggingface.co/neuphonic/neucodec).

> [!NOTE]
> We have an on-going discussion about finetuning [here](https://github.com/neuphonic/neutts-air/issues/7) where some users have reported success with finetuning using the example script.

# Finetuning on your own dataset

You can prepare your own dataset by following these steps:

1. Encode your audio files using the [NeuCodec](https://huggingface.co/neuphonic/neucodec) model into a format similar to the [Emilia-YODAS dataset](https://huggingface.co/datasets/neuphonic/emilia-yodas-english-neucodec).
2. Setup your configuration file similar to the [example config](/examples/finetune_config.yaml).
3. Check and modify the phonemizer and the tokenizer in the script such that they suit your dataset/task. See [the phonemizer documentation](https://bootphon.github.io/phonemizer/api_reference.html#phonemizer.backend.espeak.espeak.EspeakBackend) for phonemizer arguments.
4. Run the finetuning script with your dataset and configuration file. To do this, navigate to the base directory of your cloned repo in the terminal and run:

    ```bash
    python examples/finetune.py examples/finetune_config.yaml
    ```

    replacing the argument with the path to your own config file if needed.

# Finetuning config

An example finetuning config lives in `examples/finetune_config.yaml`.

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
