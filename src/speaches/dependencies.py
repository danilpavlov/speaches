"""
Модуль для определения зависимостей, используемых в приложении FastAPI.
"""

from functools import lru_cache
import logging
from typing import Annotated

import av.error
from fastapi import (
    Depends,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper.audio import decode_audio
from httpx import ASGITransport, AsyncClient
from numpy import float32
from numpy.typing import NDArray
from openai import AsyncOpenAI
from openai.resources.audio import AsyncSpeech, AsyncTranscriptions
from openai.resources.chat.completions import AsyncCompletions

#from pyannote.audio import Pipeline
import torch

from speaches.config import Config
from speaches.model_manager import KokoroModelManager, PiperModelManager, WhisperModelManager

import os
from pathlib import Path


logger = logging.getLogger(__name__)

# NOTE: `get_config` is called directly instead of using sub-dependencies so that these functions could be used outside of `FastAPI`  # noqa: E501


# https://fastapi.tiangolo.com/advanced/settings/?h=setti#creating-the-settings-only-once-with-lru_cache
# WARN: Any new module that ends up calling this function directly (not through `FastAPI` dependency injection) should be patched in `tests/conftest.py`  # noqa: E501
@lru_cache
def get_config() -> Config:
    """
    Description
        Возвращает конфигурацию приложения.

    Returns:
        Объект Config с настройками приложения.
    """
    return Config()


ConfigDependency = Annotated[Config, Depends(get_config)]


@lru_cache
def get_model_manager() -> WhisperModelManager:
    """
    Description
        Возвращает менеджер моделей Whisper.

    Returns:
        Объект WhisperModelManager.
    """
    config = get_config()
    return WhisperModelManager(config.whisper)


ModelManagerDependency = Annotated[WhisperModelManager, Depends(get_model_manager)]


@lru_cache
def get_piper_model_manager() -> PiperModelManager:
    """
    Description
        Возвращает менеджер моделей Piper.

    Returns:
        Объект PiperModelManager.
    """
    config = get_config()
    return PiperModelManager(config.whisper.ttl, config.enable_dynamic_loading)  # HACK: should have its own config


PiperModelManagerDependency = Annotated[PiperModelManager, Depends(get_piper_model_manager)]


@lru_cache
def get_kokoro_model_manager() -> KokoroModelManager:
    """
    Description
        Возвращает менеджер моделей Kokoro.

    Returns:
        Объект KokoroModelManager.
    """
    config = get_config()
    return KokoroModelManager(config.whisper.ttl, config.enable_dynamic_loading)  # HACK: should have its own config


KokoroModelManagerDependency = Annotated[KokoroModelManager, Depends(get_kokoro_model_manager)]

security = HTTPBearer()


async def verify_api_key(
    config: ConfigDependency, credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> None:
    """
    Description
        Проверяет API ключ.

    Args:
        config: Конфигурация.
        credentials: Учетные данные авторизации.

    Raises:
        HTTPException: Если API ключ неверен.
    """
    if credentials.credentials != config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


ApiKeyDependency = Depends(verify_api_key)


def audio_file_dependency(
    file: Annotated[UploadFile, Form()],
) -> NDArray[float32]:
    """
    Description
        Декодирует аудио файл.

    Args:
        file: Загруженный аудио файл.

    Returns:
        Массив аудио данных.

    Raises:
        HTTPException: Если не удалось декодировать аудио файл.
    """
    try:
        audio = decode_audio(file.file)
    except av.error.InvalidDataError as e:
        raise HTTPException(
            status_code=415,
            detail="Failed to decode audio. The provided file type is not supported.",
        ) from e
    except av.error.ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode audio. The provided file is likely empty.",
        ) from e
    except Exception as e:
        logger.exception(
            "Failed to decode audio. This is likely a bug. Please create an issue at https://github.com/speaches-ai/speaches/issues/new."
        )
        raise HTTPException(status_code=500, detail="Failed to decode audio.") from e
    else:
        return audio  # pyright: ignore reportReturnType


AudioFileDependency = Annotated[NDArray[float32], Depends(audio_file_dependency)]


@lru_cache
def get_completion_client() -> AsyncCompletions:
    """
    Description
        Возвращает клиент для завершения чатов.

    Returns:
        Объект AsyncCompletions.
    """
    config = get_config()
    oai_client = AsyncOpenAI(base_url=config.chat_completion_base_url, api_key=config.chat_completion_api_key)
    return oai_client.chat.completions


CompletionClientDependency = Annotated[AsyncCompletions, Depends(get_completion_client)]


@lru_cache
def get_speech_client() -> AsyncSpeech:
    """
    Description
        Возвращает клиент для синтеза речи.

    Returns:
        Объект AsyncSpeech.
    """
    config = get_config()
    if config.speech_base_url is None:
        from speaches.routers.speech import (
            router as speech_router,
        )

        http_client = AsyncClient(
            transport=ASGITransport(speech_router), base_url="http://test/v1"
        )  # NOTE: "test" can be replaced with any other value
        oai_client = AsyncOpenAI(http_client=http_client, api_key=config.speech_api_key)
    else:
        oai_client = AsyncOpenAI(base_url=config.speech_base_url, api_key=config.speech_api_key)
    return oai_client.audio.speech


SpeechClientDependency = Annotated[AsyncSpeech, Depends(get_speech_client)]


@lru_cache
def get_transcription_client() -> AsyncTranscriptions:
    """
    Description
        Возвращает клиент для транскрипции аудио.

    Returns:
        Объект AsyncTranscriptions.
    """
    config = get_config()
    if config.transcription_base_url is None:
        from speaches.routers.stt import (
            router as stt_router,
        )

        http_client = AsyncClient(
            transport=ASGITransport(stt_router), base_url="http://test/v1"
        )  # NOTE: "test" can be replaced with any other value

        oai_client = AsyncOpenAI(http_client=http_client, api_key=config.transcription_api_key)
    else:
        oai_client = AsyncOpenAI(base_url=config.transcription_base_url, api_key=config.transcription_api_key)
    return oai_client.audio.transcriptions


TranscriptionClientDependency = Annotated[AsyncTranscriptions, Depends(get_transcription_client)]


# @lru_cache
# def get_diarization_model() -> Pipeline:
#     """
#     Description
#         Возвращает модель диаризации.

#     Note:
#         Немного неочевидный трюк с вызовом функции с аргументом по умолчанию, чтобы избежать проблем с множественной загрузкой модели.

#     Returns:
#         Объект Pipeline с моделью диаризации.
#     """
#     def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
#         path_to_config = Path(path_to_config)

#         print(f"Loading pyannote pipeline from {path_to_config}...")
#         # the paths in the config are relative to the current working directory
#         # so we need to change the working directory to the model path
#         # and then change it back

#         cwd = Path.cwd().resolve()  # store current working directory

#         # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
#         cd_to = path_to_config.parent.parent.resolve()

#         print(f"Changing working directory to {cd_to}")
#         os.chdir(cd_to)

#         pipeline = Pipeline.from_pretrained(path_to_config)

#         print(f"Changing working directory back to {cwd}")
#         os.chdir(cwd)

#         return pipeline
#     config = get_config()  # HACK
#     diarization_pipeline = load_pipeline_from_pretrained('./pyannote_diarization_config.yaml')
#     logger.info(f'Загружаем модель диаризации на устройство: {config.diarization.device}')
#     if config.diarization.device != 'cpu':
#         diarization_pipeline = diarization_pipeline.to(torch.device(config.diarization.device))
#     assert diarization_pipeline, "Не удалось загрузить модель диаризации!"
#     return diarization_pipeline

# DiarizationModelDependency = Annotated[Pipeline, Depends(get_diarization_model)]


def diarization_audio_file_dependency(
    file: Annotated[UploadFile, Form()],
) -> NDArray[float32]:
    """
    Description
        Декодирует аудио файл.

    Args:
        file: Загруженный аудио файл.

    Returns:
        Массив аудио данных.

    Raises:
        HTTPException: Если не удалось декодировать аудио файл.
    """
    try:
        audio = decode_audio(file.file)
        filename = file.filename
    except av.error.InvalidDataError as e:
        raise HTTPException(
            status_code=415,
            detail="Failed to decode audio. The provided file type is not supported.",
        ) from e
    except av.error.ValueError as e:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode audio. The provided file is likely empty.",
        ) from e
    except Exception as e:
        logger.exception(
            "Failed to decode audio. This is likely a bug. Please create an issue at https://github.com/speaches-ai/speaches/issues/new."
        )
        raise HTTPException(status_code=500, detail="Failed to decode audio.") from e
    else:
        return {'audio': audio, 'filename': filename}  # pyright: ignore reportReturnType


DiarizationAudioFileDependency = Annotated[dict[str, NDArray[float32] | str | str], Depends(diarization_audio_file_dependency)]