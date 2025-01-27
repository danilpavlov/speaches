"""
Модуль для обработки текстовых данных, включая сегментацию и форматирование транскрипций.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from speaches.api_types import TranscriptionSegment, TranscriptionWord


class Transcription:
    """
    Класс для представления транскрипции.
    """
    def __init__(self, words: list[TranscriptionWord] = []) -> None:
        self.words: list[TranscriptionWord] = []
        self.extend(words)

    @property
    def text(self) -> str:
        return " ".join(word.word for word in self.words).strip()

    @property
    def start(self) -> float:
        return self.words[0].start if len(self.words) > 0 else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if len(self.words) > 0 else 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def after(self, seconds: float) -> Transcription:
        """
        Description
            Возвращает транскрипцию после заданного времени.

        Args:
            seconds: Время в секундах.

        Returns:
            Объект Transcription с транскрипцией после заданного времени.
        """
        return Transcription(words=[word for word in self.words if word.start > seconds])

    def extend(self, words: list[TranscriptionWord]) -> None:
        """
        Description
            Расширяет транскрипцию новыми словами.

        Args:
            words: Список новых слов.
        """
        self._ensure_no_word_overlap(words)
        self.words.extend(words)

    def _ensure_no_word_overlap(self, words: list[TranscriptionWord]) -> None:
        from speaches.dependencies import get_config  # HACK: avoid circular import

        config = get_config()  # HACK
        if len(self.words) > 0 and len(words) > 0:
            if words[0].start + config.word_timestamp_error_margin <= self.words[-1].end:
                raise ValueError(
                    f"Words overlap: {self.words[-1]} and {words[0]}. Error margin: {config.word_timestamp_error_margin}"  # noqa: E501
                )
        for i in range(1, len(words)):
            if words[i].start + config.word_timestamp_error_margin <= words[i - 1].end:
                raise ValueError(f"Words overlap: {words[i - 1]} and {words[i]}. All words: {words}")


def is_eos(text: str) -> bool:
    """
    Description
        Проверяет, является ли текст концом предложения.

    Args:
        text: Текст для проверки.

    Returns:
        Булево значение, указывающее на конец предложения.
    """
    if (text.endswith("...")):
        return False
    return any(text.endswith(punctuation_symbol) for punctuation_symbol in ".?!")


def to_full_sentences(words: list[TranscriptionWord]) -> list[list[TranscriptionWord]]:
    """
    Description
        Преобразует список слов в список полных предложений.

    Args:
        words: Список слов.

    Returns:
        Список списков слов, представляющих полные предложения.
    """
    sentences: list[list[TranscriptionWord]] = [[]]
    for word in words:
        sentences[-1].append(word)
        if is_eos(word.word):
            sentences.append([])
    if len(sentences[-1]) == 0 or not is_eos(sentences[-1][-1].word):
        sentences.pop()
    return sentences


def word_to_text(words: list[TranscriptionWord]) -> str:
    """
    Description
        Преобразует список слов в текст.

    Args:
        words: Список слов.

    Returns:
        Строка текста.
    """
    return "".join(word.word for word in words)


def words_to_text_w_ts(words: list[TranscriptionWord]) -> str:
    """
    Description
        Преобразует список слов в текст с временными метками.

    Args:
        words: Список слов.

    Returns:
        Строка текста с временными метками.
    """
    return "".join(f"{word.word}({word.start:.2f}-{word.end:.2f})" for word in words)


def segments_to_text(segments: Iterable[TranscriptionSegment]) -> str:
    """
    Description
        Преобразует сегменты транскрипции в текст.

    Args:
        segments: Сегменты транскрипции.

    Returns:
        Строка текста.
    """
    return "".join(segment.text for segment in segments).strip()


def srt_format_timestamp(ts: float) -> str:
    """
    Description
        Форматирует временную метку в формате SRT.

    Args:
        ts: Временная метка в секундах.

    Returns:
        Строка временной метки в формате SRT.
    """
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def vtt_format_timestamp(ts: float) -> str:
    """
    Description
        Форматирует временную метку в формате VTT.

    Args:
        ts: Временная метка в секундах.

    Returns:
        Строка временной метки в формате VTT.
    """
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


def segments_to_vtt(segment: TranscriptionSegment, i: int) -> str:
    """
    Description
        Преобразует сегмент транскрипции в формат VTT.

    Args:
        segment: Сегмент транскрипции.
        i: Индекс сегмента.

    Returns:
        Строка сегмента в формате VTT.
    """
    start = segment.start if i > 0 else 0.0
    result = f"{vtt_format_timestamp(start)} --> {vtt_format_timestamp(segment.end)}\n{segment.text}\n\n"

    if i == 0:
        return f"WEBVTT\n\n{result}"
    else:
        return result


def segments_to_srt(segment: TranscriptionSegment, i: int) -> str:
    """
    Description
        Преобразует сегмент транскрипции в формат SRT.

    Args:
        segment: Сегмент транскрипции.
        i: Индекс сегмента.

    Returns:
        Строка сегмента в формате SRT.
    """
    return f"{i + 1}\n{srt_format_timestamp(segment.start)} --> {srt_format_timestamp(segment.end)}\n{segment.text}\n\n"


def canonicalize_word(text: str) -> str:
    """
    Description
        Приводит слово к каноническому виду.

    Args:
        text: Слово для канонизации.

    Returns:
        Каноническое слово.
    """
    text = text.lower()
    text = re.sub(r"[^a-z]", "", text)
    return text.lower().strip().strip(".,?!")


def common_prefix(a: list[TranscriptionWord], b: list[TranscriptionWord]) -> list[TranscriptionWord]:
    """
    Description
        Находит общий префикс в двух списках слов.

    Args:
        a: Первый список слов.
        b: Второй список слов.

    Returns:
        Список слов, представляющих общий префикс.
    """
    i = 0
    while i < len(a) and i < len(b) and canonicalize_word(a[i].word) == canonicalize_word(b[i].word):
        i += 1
    return a[:i]
