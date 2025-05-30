"""
Модуль для автоматического распознавания речи (ASR) с использованием модели Faster Whisper.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from speaches.api_types import TranscriptionSegment, TranscriptionWord
from speaches.text_utils import Transcription

if TYPE_CHECKING:
    from faster_whisper import transcribe

    from speaches.audio import Audio

logger = logging.getLogger(__name__)


class FasterWhisperASR:
    """
    Класс для автоматического распознавания речи с использованием модели Faster Whisper.
    """
    def __init__(
        self,
        whisper: transcribe.WhisperModel,
        **kwargs,
    ) -> None:
        self.whisper = whisper
        self.transcribe_opts = kwargs

    def _transcribe(
        self,
        audio: Audio,
        prompt: str | None = None,
    ) -> tuple[Transcription, transcribe.TranscriptionInfo]:
        """
        Description
            Выполняет распознавание речи.

        Args:
            audio: Аудио данные.
            prompt: Начальная подсказка.

        Returns:
            Кортеж с объектом Transcription и информацией о транскрипции.
        """
        start = time.perf_counter()
        segments, transcription_info = self.whisper.transcribe(
            audio.data,
            initial_prompt=prompt,
            word_timestamps=True,
            **self.transcribe_opts,
        )
        segments = TranscriptionSegment.from_faster_whisper_segments(segments)
        words = TranscriptionWord.from_segments(segments)
        for word in words:
            word.offset(audio.start)
        transcription = Transcription(words)
        end = time.perf_counter()
        logger.info(
            f"Transcribed {audio} in {end - start:.2f} seconds. Prompt: {prompt}. Transcription: {transcription.text}"
        )
        return (transcription, transcription_info)

    async def transcribe(
        self,
        audio: Audio,
        prompt: str | None = None,
    ) -> tuple[Transcription, transcribe.TranscriptionInfo]:
        """
        Description
            Асинхронно выполняет распознавание речи.

        Args:
            audio: Аудио данные.
            prompt: Начальная подсказка.

        Returns:
            Кортеж с объектом Transcription и информацией о транскрипции.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._transcribe,
            audio,
            prompt,
        )
