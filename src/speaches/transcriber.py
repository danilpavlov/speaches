"""
Модуль для транскрипции аудио данных с использованием модели Faster Whisper.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from speaches.audio import Audio, AudioStream
from speaches.text_utils import Transcription, common_prefix, to_full_sentences, word_to_text

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from speaches.api_types import TranscriptionWord
    from speaches.asr import FasterWhisperASR

logger = logging.getLogger(__name__)


class LocalAgreement:
    """
    Класс для локального согласования транскрипций.
    """
    def __init__(self) -> None:
        self.unconfirmed = Transcription()

    def merge(self, confirmed: Transcription, incoming: Transcription) -> list[TranscriptionWord]:
        """
        Description
            Объединяет подтвержденные и входящие транскрипции.

        Args:
            confirmed: Подтвержденная транскрипция.
            incoming: Входящая транскрипция.

        Returns:
            Список слов из общего префикса.
        """
        incoming = incoming.after(confirmed.end - 0.1)
        prefix = common_prefix(incoming.words, self.unconfirmed.words)
        logger.debug(f"Confirmed: {confirmed.text}")
        logger.debug(f"Unconfirmed: {self.unconfirmed.text}")
        logger.debug(f"Incoming: {incoming.text}")

        if len(incoming.words) > len(prefix):
            self.unconfirmed = Transcription(incoming.words[len(prefix) :])
        else:
            self.unconfirmed = Transcription()

        return prefix


# TODO: needs a better name
def needs_audio_after(confirmed: Transcription) -> float:
    """
    Description
        Определяет, сколько аудио данных нужно после подтвержденной транскрипции.

    Args:
        confirmed: Подтвержденная транскрипция.

    Returns:
        Время в секундах.
    """
    full_sentences = to_full_sentences(confirmed.words)
    return full_sentences[-1][-1].end if len(full_sentences) > 0 else 0.0


def prompt(confirmed: Transcription) -> str | None:
    """
    Description
        Возвращает подсказку для следующей транскрипции.

    Args:
        confirmed: Подтвержденная транскрипция.

    Returns:
        Строка с подсказкой или None.
    """
    sentences = to_full_sentences(confirmed.words)
    return word_to_text(sentences[-1]) if len(sentences) > 0 else None


async def audio_transcriber(
    asr: FasterWhisperASR,
    audio_stream: AudioStream,
    min_duration: float,
) -> AsyncGenerator[Transcription, None]:
    """
    Description
        Асинхронно транскрибирует аудио данные из потока.

    Args:
        asr: Модель для автоматического распознавания речи.
        audio_stream: Поток аудио данных.
        min_duration: Минимальная длительность чанка в секундах.

    Returns:
        Асинхронный генератор транскрипций.
    """
    local_agreement = LocalAgreement()
    full_audio = Audio()
    confirmed = Transcription()
    async for chunk in audio_stream.chunks(min_duration):
        full_audio.extend(chunk)
        audio = full_audio.after(needs_audio_after(confirmed))
        transcription, _ = await asr.transcribe(audio, prompt(confirmed))
        new_words = local_agreement.merge(confirmed, transcription)
        if len(new_words) > 0:
            confirmed.extend(new_words)
            yield confirmed
    logger.debug("Flushing...")
    confirmed.extend(local_agreement.unconfirmed.words)
    yield confirmed
    logger.info("Audio transcriber finished")
