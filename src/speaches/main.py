from __future__ import annotations

import logging
import platform

import uvicorn

from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware

from speaches.dependencies import ApiKeyDependency, get_config
from speaches.logger import setup_logger
from speaches.routers.misc import (
    router as misc_router,
)
from speaches.routers.models import (
    router as models_router,
)
from speaches.routers.speech import (
    router as speech_router,
)
from speaches.routers.stt import (
    router as stt_router,
)
from speaches.routers.vad import (
    router as vad_router,
)
from speaches.routers.diarization import (
    router as diarization_router,
)
from contextlib import asynccontextmanager
import os

from speaches.dependencies import get_model_manager, get_piper_model_manager, get_diarization_model

DESCRIPTION = """
# Speaches - Сервис для распознавания и синтеза речи

## 📝 Описание
Универсальный сервис для работы с аудио, включающий распознавание речи,
определение говорящих и синтез речи.

## 🚀 Основные возможности
1. Распознавание речи (STT):
   - Поддержка Whisper моделей
   - Streaming режим для real-time транскрипции
   - VAD фильтрация тишины
   - Поддержка hotwords для улучшения точности
   - Различные форматы вывода (text, json, vtt, srt)

2. Диаризация (определение говорящих):
   - Базовая диаризация (/diarize)
   - Расширенная диаризация с транскрипцией (/v1/audio/diarization)
   - Настройка количества говорящих
   - Временные метки для каждого говорящего

3. Синтез речи (TTS):
   - Поддержка моделей Kokoro и Piper
   - Различные языки и голоса
   - Настраиваемые параметры синтеза
   - Streaming передача аудио
   - Форматы: mp3, wav, ogg

## 🛠️ Основные эндпоинты
1. Speech-to-Text (STT):
   - POST /v1/audio/transcriptions - Распознавание речи
   
2. Диаризация:
   - POST /diarize - Базовая диаризация
   - POST /v1/audio/diarization - Расширенная диаризация
   
3. Text-to-Speech (TTS):
   - POST /v1/audio/speech - Синтез речи

## 📦 Поддерживаемые модели
- Whisper (STT): faster-whisper-large-v3-turbo
- Pyannote (Диаризация): pyannote/speaker-diarization-3.1
- TTS: rhasspy/piper-voices

## ⚙️ Конфигурация
- Поддержка GPU для ускорения
- Настройка размера батча
- Кастомизация параметров моделей
- Управление динамической загрузки модели

## 📋 Примеры использования
### 1. Диаризация аудио
```bash
curl -X POST "http://localhost:8000/v1/audio/diarization" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@audio.wav" \
        -F "model=base" \
        -F "language=ru" \
        -F "response_format=verbose_json" \
        -F "num_speakers=2" \
        -F "timestamp_granularities=segment" \
        -F "timestamp_granularities=word"
```
Ожидаем:
```json
[
  {
    "id": 1,
    "start": 0.0,
    "end": 2.0,
    "text": " Привет! Меня зовут Пайпер!",
    "speaker": "SPEAKER_00"
  }
]
```

### 2. Точечная диаризация
```bash
curl -X POST http://localhost:8000/diarize \
        -F "audio=@audio.wav" \
        -F "num_speakers=2"
```
Ожидаем:
```json
{
  "diarization_segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.03096875,
      "end": 1.9209687500000001
    }
  ],
  "success": true,
  "error": null
}
```

### 3. STT
```bash
curl http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav"
```
Ожидаем:
```json
{
  "text": "Привет, меня зовут Пайпер."
}
```

### 4. TTS
```bash
curl http://localhost:8000/v1/audio/speech \
        -H "Content-Type: application/json" \
        -d '{
  "model": "rhasspy/piper-voices",
  "input": "Привет, меня зовут Пайпер!",
  "voice": "ru_RU-denis-medium",
  "response_format": "wav",
  "speed": 1,
  "sample_rate": 8000
}' \
        --output audio.wav
```
Ожидаем: audio.wav

"""

# https://swagger.io/docs/specification/v3_0/grouping-operations-with-tags/
# https://fastapi.tiangolo.com/tutorial/metadata/#metadata-for-tags
TAGS_METADATA = [
    {"name": "automatic-speech-recognition"},
    {"name": "speech-to-text"},
    {"name": "models"},
    {"name": "diagnostic"},
    {
        "name": "experimental",
        "description": "Not meant for public use yet. May change or be removed at any time.",
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Starting up')
    _ = get_diarization_model()
    model_managers = {
        os.getenv('WHISPER__MODEL', None): get_model_manager(), 
        "ru_RU-denis-medium": get_piper_model_manager()
    }
    for name, model in model_managers.items():
        print('Loading: ', type(model).__name__)
        if name:
            with model.load_model(name) as model_instance:
                pass
    yield
    print('Shutting down')
    


def create_app() -> FastAPI:
    config = get_config()  # HACK
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    if platform.machine() == "x86_64":
        logger.warning("`POST /v1/audio/speech` with `model=rhasspy/piper-voices` is only supported on x86_64 machines")

    dependencies = []
    if config.api_key is not None:
        dependencies.append(ApiKeyDependency)

    app = FastAPI(
        dependencies=dependencies, 
        openapi_tags=TAGS_METADATA, 
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    routers = [
            stt_router, models_router, 
            misc_router, speech_router, 
            vad_router, diarization_router
        ]
    for router in routers:
        app.include_router(router)

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if config.enable_ui:
        import gradio as gr

        from speaches.gradio_app import create_gradio_demo

        app = gr.mount_gradio_app(app, create_gradio_demo(config), path="/")

    return app


if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        app=create_app,
        host=config.host,
        port=config.port,
        factory=True,
    )