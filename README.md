> [!NOTE]
> This project was previously named `faster-whisper-server`. I've decided to change the name from `faster-whisper-server`, as the project has evolved to support more than just transcription.


# Getting Started


# Установка весов моделей:

## 1 Вариант: через huggingface-cli

```bash
pip install 'huggingface_hub[cli]'
huggingface-cli login

# Если на этапе логина: PermissionError
chmod 777 ~/.cache/huggingface
```

```bash
# Этот мув нужен, так как speaches использует именно кэш моделей. (будет искать с models--*)
export HF_HOME=$(pwd)/cache/huggingface
export TRANSFORMERS_CACHE=$(pwd)/cache/huggingface

# Скачивание весов
huggingface-cli download h2oai/faster-whisper-large-v3-turbo
huggingface-cli download pyannote/speaker-diarization-3.1
# Загружаем только RU и EN голоса:
for f in "voices.json" "ru/*" "en/*" "_script/*" "README.md"; do huggingface-cli download rhasspy/piper-voices --include "$f"; done

# Возвращаем директорию в исходное состояние
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface
```


## 2 Вариант: через докер образ с кэшем моделей

```bash
# Через докер образ с кэшем моделей:
docker run --rm -it -v $(pwd)/cache/huggingface/hub:/cache eyeonyou/speaches-cache:0.0.2 mv ./models--* /cache
```

```bash
uv sync --frozen --compile-bytecode --extra ui
```


# Speaches

`speaches` is an OpenAI API-compatible server supporting streaming transcription, translation, and speech generation. Speach-to-Text is powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and for Text-to-Speech [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) are used. This project aims to be Ollama, but for TTS/STT models.

Try it out on the [HuggingFace Space](https://huggingface.co/spaces/speaches-ai/speaches)

See the documentation for installation instructions and usage: [https://speaches-ai.github.io/speaches/](https://speaches-ai.github.io/speaches/)

## Features:

- GPU and CPU support.
- [Deployable via Docker Compose / Docker](https://speaches-ai.github.io/speaches/installation/)
- [Highly configurable](https://speaches-ai.github.io/speaches/configuration/)
- OpenAI API compatible. All tools and SDKs that work with OpenAI's API should work with `speaches`.
- Streaming support (transcription is sent via SSE as the audio is transcribed. You don't need to wait for the audio to fully be transcribed before receiving it).

  - LocalAgreement2 ([paper](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) | [original implementation](https://github.com/ufal/whisper_streaming)) algorithm is used for live transcription.

- Live transcription support (audio is sent via websocket as it's generated).
- Dynamic model loading / offloading. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.
- Text-to-Speech via `kokoro`(Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models.
- [Coming soon](https://github.com/speaches-ai/speaches/issues/231): Audio generation (chat completions endpoint) | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)
  - Generate a spoken audio summary of a body of text (text in, audio out)
  - Perform sentiment analysis on a recording (audio in, text out)
  - Async speech to speech interactions with a model (audio in, audio out)
- [Coming soon](https://github.com/speaches-ai/speaches/issues/115): Realtime API | [OpenAI Documentation](https://platform.openai.com/docs/guides/realtime)

Please create an issue if you find a bug, have a question, or a feature suggestion.

## Demo

### Streaming Transcription

TODO

### Speech Generation

https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b


### Live Transcription (using WebSockets)

https://github.com/fedirz/faster-whisper-server/assets/76551385/e334c124-af61-41d4-839c-874be150598f
