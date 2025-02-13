"""
Модуль для сопоставления говорящих с сегментами транскрипции.
"""

from dataclasses import dataclass, asdict
import json
from typing import List, Optional

@dataclass
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

@dataclass
class DiarizationSegment:
    speaker: str
    start: float
    end: float
    
def merge_speakers(segments: list[dict]) -> list[dict]:
    merged = []
    current_speaker = None
    current_segment = None
    
    for segment in segments:
        if current_speaker == segment['speaker']:
            # Если спикер тот же, расширяем текущий сегмент
            current_segment['end'] = segment['end']
            current_segment['text'] += ' ' + segment['text']
        else:
            # Если спикер изменился, добавляем текущий сегмент и начинаем новый
            if current_segment:
                merged.append(current_segment)
            current_segment = segment.copy()
            current_speaker = segment['speaker']
    
    # Добавляем последний сегмент
    if current_segment:
        merged.append(current_segment)
    
    return merged

def calculate_overlap(trans_seg: TranscriptionSegment, diar_seg: DiarizationSegment) -> float:
    """
    Description
        Вычисляет временное перекрытие между сегментом транскрипции и сегментом диаризации.

    Args:
        trans_seg: Сегмент транскрипции.
        diar_seg: Сегмент диаризации.

    Returns:
        Величина перекрытия во времени.
    """
    overlap_start = max(trans_seg.start, diar_seg.start)
    overlap_end = min(trans_seg.end, diar_seg.end)
    return max(0, overlap_end - overlap_start)

def map_speakers_to_segments(
    transcription_segments: List[TranscriptionSegment],
    diarization_segments: List[DiarizationSegment]
) -> str:
    """
    Description
        Сопоставляет говорящих с сегментами транскрипции и возвращает JSON строку.

    Args:
        transcription_segments: Список сегментов транскрипции.
        diarization_segments: Список сегментов диаризации.

    Returns:
        JSON строка с результатами сопоставления.
    """
    result = []
    
    for trans_seg in transcription_segments:
        max_overlap = 0
        best_speaker = None
        
        # Find diarization segment with maximum overlap
        for diar_seg in diarization_segments:
            overlap = calculate_overlap(trans_seg, diar_seg)
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diar_seg.speaker
        
        # Create new segment with assigned speaker
        new_segment = TranscriptionSegment(
            id=trans_seg.id,
            start=trans_seg.start,
            end=trans_seg.end,
            text=trans_seg.text,
            speaker=best_speaker
        )
        result.append(new_segment)
    
    # Convert the result to a list of dictionaries
    result_json = [asdict(segment) for segment in result]
    
    result_json = merge_speakers(result_json)
    
    # Return JSON string with proper encoding for Cyrillic characters
    return json.dumps(result_json, ensure_ascii=False, indent=2)