"""
Модуль для диаризации аудио (определение говорящих).

Предоставляет два основных эндпоинта:
- /diarize: Простая диаризация аудио
- /v1/audio/diarization: Расширенная диаризация в формате OpenAI

Использует pyannote.audio для определения говорящих и faster-whisper для транскрипции.
"""
from fastapi import APIRouter, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import StreamingResponse, Response
from speaches.dependencies import ConfigDependency, AudioFileDependency, ModelManagerDependency, DiarizationModelDependency
from io import BytesIO
import logging

from pydantic import BaseModel, Field
from typing import Optional, List, Annotated
from speaches.api_types import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
    TimestampGranularities,
    TranscriptionSegment
)
import asyncio
from faster_whisper.transcribe import BatchedInferencePipeline
from speaches.map_speakers import map_speakers_to_segments, DiarizationSegment
from speaches.config import Task, ResponseFormat
from speaches.routers.stt import ModelName, Language, get_timestamp_granularities

import torchaudio
import torch

logger = logging.getLogger(__name__)
router = APIRouter(tags=['Диаризация'])

class DiarizationResponse(BaseModel):
    diarization_segments: List[dict] = Field(default_factory=list)
    success: bool = Field(default=False)
    error: Optional[str] = Field(default=None)


async def diarize_audio(
    config: ConfigDependency,
    diarization_manager: DiarizationModelDependency,
    audio: UploadFile | BytesIO = File(...),
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> DiarizationResponse:
    """
    # Description
        Выполняет диаризацию аудио файла с помощью pyannote.audio.

    ## Args:
        config: Конфигурация приложения
        audio: Аудио файл (UploadFile или BytesIO)
        num_speakers: Точное количество говорящих
        min_speakers: Минимальное количество говорящих
        max_speakers: Максимальное количество говорящих

    ## Returns:
        DiarizationResponse с сегментами диаризации

    ## Raises:
        Exception: При ошибке диаризации

    #### Examples:
        >>> response = await diarize_audio(
        ...     config=config,
        ...     audio=audio_file,
        ...     num_speakers=2
        ... )
        >>> print(response.diarization_segments)
        [{'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 1.5}, ...]
    """
    try:
        logger.info(audio)
        # Perform diarization with the properly formatted audio input
        # Read the audio file
        audio_stream = audio if isinstance(audio, BytesIO) else BytesIO(await audio.read())
        audio_stream.seek(0)  # Ensure we're at start of stream
        
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_stream)
        diarization = diarization_manager(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
            min_speakers=min_speakers or config.diarization.min_speakers,
            max_speakers=max_speakers or config.diarization.max_speakers,
        )
        
        # Convert diarization output to the expected format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": float(turn.start),
                "end": float(turn.end)
            })
        return DiarizationResponse(
            diarization_segments=segments,
            success=True
        )
            
    except Exception as e:
        logger.exception("Diarization failed")
        return DiarizationResponse(
            diarization_segments=[],
            success=False,
            error=str(e)
        )
        
@router.post(
    '/diarize', 
    response_model=DiarizationResponse,
    summary="Точечная диаризация аудиофайла")
async def diarize(
    config: ConfigDependency,
    diarization_manager: DiarizationModelDependency,
    audio: UploadFile = File(...),
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None
):
    """
    # Description
        Эндпоинт для базовой диаризации аудио. Определяет говорящих в аудио файле.

    #### Features:
        - Определение количества говорящих
        - Временные метки для каждого говорящего
        - Настраиваемые параметры диаризации
        
    ## Args:
        config: Конфигурация приложения
        audio: Аудио файл
        num_speakers: Точное количество говорящих
        min_speakers: Минимальное количество говорящих
        max_speakers: Максимальное количество говорящих

    ## Returns:
        DiarizationResponse с результатами диаризации

    #### Examples:
        >>> with open('audio.wav', 'rb') as f:
        ...     file = UploadFile(f)
        ...     response = await diarize(
        ...         config=config,
        ...         audio=file,
        ...         num_speakers=2
        ...     )
        >>> print(response.diarization_segments)
        [{'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 1.5}, ...]
    """
    print(f'DIARIZATION MANAGER: {diarization_manager}')
    return await diarize_audio(
        config, diarization_manager, audio, num_speakers, min_speakers, max_speakers
    )


@router.post(
    "/v1/audio/diarization",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
    summary="Диаризация аудиофайла (OpenAI формат)")
def diarize_file(
    config: ConfigDependency,
    diarization_manager: DiarizationModelDependency,
    model_manager: ModelManagerDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelName | None, Form()] = None,
    language: Annotated[Language | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat | None, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        TimestampGranularities,
        # WARN: `alias` doesn't actually work.
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    # stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    vad_filter: Annotated[bool, Form()] = False,
    num_speakers: Annotated[int | None, Form()] = None,
) -> Response | StreamingResponse:
    """
    # Description
        Комбинирует распознавание речи и диаризацию в формате OpenAI.
        
    #### Features:
        - Распознавание речи с помощью Whisper
        - Диаризация с помощью pyannote.audio
        - Поддержка hotwords и VAD
        - Различные форматы ответа

    ## Args:
        config: Конфигурация приложения
        model_manager: Менеджер моделей
        request: FastAPI запрос
        audio: Аудио файл
        model: Модель Whisper
        language: Язык аудио
        prompt: Промпт для улучшения распознавания
        response_format: Формат ответа
        temperature: Температура сэмплирования
        timestamp_granularities: Гранулярность временных меток
        hotwords: Ключевые слова
        vad_filter: Использовать VAD фильтр
        num_speakers: Количество говорящих

    ## Returns:
        Response или StreamingResponse с результатами диаризации и транскрипции

    ## Raises:
        HTTPException: При ошибках обработки

    #### Examples:
        >>> response = await diarize_file(
        ...     config=config,
        ...     model_manager=model_manager,
        ...     request=request,
        ...     audio=audio_file,
        ...     model="whisper-1",
        ...     language="ru",
        ...     response_format="verbose_json",
        ...     num_speakers=2
        ... )
        >>> print(response)
        `{"segments": [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00", "text": "Привет, мир!"}, ...]}`
    """
    if model is None:
        model = config.whisper.model
    if language is None:
        language = config.default_language
    if response_format is None:
        response_format = config.default_response_format
    timestamp_granularities =  asyncio.run(get_timestamp_granularities(request))
    if timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES and response_format != ResponseFormat.VERBOSE_JSON:
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to `verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities."  # noqa: E501
        )
    with model_manager.load_model(model) as whisper:
        whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        segments, transcription_info = whisper_model.transcribe(
            audio,
            task=Task.TRANSCRIBE,
            language=language,
            initial_prompt=prompt,
            word_timestamps="word" in timestamp_granularities,
            temperature=temperature,
            vad_filter=vad_filter,
            hotwords=hotwords,
        )
        # segments = TranscriptionSegment.from_faster_whisper_segments(segments)
        transcription_segments = TranscriptionSegment.from_faster_whisper_segments(segments)

    params = {'num_speakers': num_speakers} if num_speakers else {}
    
    try:
        # Convert numpy array to bytes
        audio_bytes = BytesIO()
        
        # Save as WAV file
        torchaudio.save(
            audio_bytes,
            torch.from_numpy(audio).unsqueeze(0),  # Add channel dimension
            sample_rate=16000,  # faster-whisper uses 16kHz
            format="wav"
        )
        audio_bytes.seek(0)  # Reset buffer position
        
        # Send request with proper file formatting
        diarization_response = asyncio.run(diarize(
            config=config,
            diarization_manager=diarization_manager,
            audio=audio_bytes,
            #files=files,
            **params
        ))
        
        logger.info(diarization_response)
        if not diarization_response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Diarization failed: {diarization_response.error}"
            )
            
        # Assign speakers to segments based on time overlap
        # Convert diarization segments to proper objects
        diarization_segments = [
                DiarizationSegment(
                    speaker=seg["speaker"],
                    start=float(seg["start"]),
                    end=float(seg["end"])
                )
                for seg in diarization_response.diarization_segments
            ]
        logger.info(transcription_segments)
        logger.info(diarization_segments)
        
        # Map speakers to segments and return JSON response directly
        result = map_speakers_to_segments(transcription_segments, diarization_segments)
        return Response(
                content=result,  # result is already a JSON string from map_speakers_to_segments
                media_type="application/json"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to diarization service: {str(e)}"
        )