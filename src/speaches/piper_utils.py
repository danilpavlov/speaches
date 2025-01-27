"""
Модуль для работы с моделью Piper, включая генерацию аудио.
"""

from collections.abc import Generator
import logging
import time

from piper.voice import PiperVoice

from speaches.audio import resample_audio

logger = logging.getLogger(__name__)


def generate_audio(
    piper_tts: PiperVoice, text: str, *, speed: float = 1.0, sample_rate: int | None = None
) -> Generator[bytes, None, None]:
    """
    Description
        Генерирует аудио с использованием модели Piper.

    Args:
        piper_tts: Модель Piper.
        text: Текст для генерации аудио.
        speed: Скорость генерации.
        sample_rate: Частота дискретизации.

    Returns:
        Генератор байтов аудио данных.
    """
    if sample_rate is None:
        sample_rate = piper_tts.config.sample_rate
    start = time.perf_counter()
    for audio_bytes in piper_tts.synthesize_stream_raw(text, length_scale=1.0 / speed):
        if sample_rate != piper_tts.config.sample_rate:
            audio_bytes = resample_audio(audio_bytes, piper_tts.config.sample_rate, sample_rate)  # noqa: PLW2901
        yield audio_bytes
    logger.info(f"Generated audio for {len(text)} characters in {time.perf_counter() - start}s")
