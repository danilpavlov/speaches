"""
Модуль для диаризации аудио (определение говорящих).

Предоставляет два основных эндпоинта:
- /diarize: Простая диаризация аудио
- /v1/audio/diarization: Расширенная диаризация в формате OpenAI

Использует pyannote.audio для определения говорящих и faster-whisper для транскрипции.
"""
from fastapi import APIRouter, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import StreamingResponse, Response
from speaches.dependencies import ConfigDependency, DiarizationAudioFileDependency, ModelManagerDependency
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

from speaches.nemo_diarizer import create_basic_config, diarize as diarize_nemo

import torchaudio
import torch
import os
from speaches.hash_ import get_file_hash

logger = logging.getLogger(__name__)
router = APIRouter(tags=['Диаризация'])

class DiarizationResponse(BaseModel):
    diarization_segments: List[dict] = Field(default_factory=list)
    success: bool = Field(default=False)
    error: Optional[str] = Field(default=None)


# async def diarize_audio(
#     config: ConfigDependency,
#     #diarization_manager: DiarizationModelDependency,
#     audio: UploadFile | BytesIO = File(...),
#     num_speakers: int | None = None,
#     min_speakers: int | None = None,
#     max_speakers: int | None = None,
# ) -> DiarizationResponse:
#     """
#     # Description
#         Выполняет диаризацию аудио файла с помощью pyannote.audio.

#     ## Args:
#         config: Конфигурация приложения
#         audio: Аудио файл (UploadFile или BytesIO)
#         num_speakers: Точное количество говорящих
#         min_speakers: Минимальное количество говорящих
#         max_speakers: Максимальное количество говорящих

#     ## Returns:
#         DiarizationResponse с сегментами диаризации

#     ## Raises:
#         Exception: При ошибке диаризации

#     #### Examples:
#         >>> response = await diarize_audio(
#         ...     config=config,
#         ...     audio=audio_file,
#         ...     num_speakers=2
#         ... )
#         >>> print(response.diarization_segments)
#         [{'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 1.5}, ...]
#     """
#     try:
#         logger.info(audio)
#         # Perform diarization with the properly formatted audio input
#         # Read the audio file
#         audio_stream = audio if isinstance(audio, BytesIO) else BytesIO(await audio.read())
#         audio_stream.seek(0)  # Ensure we're at start of stream
        
#         # Load audio using torchaudio
#         waveform, sample_rate = torchaudio.load(audio_stream)
#         diarization = diarization_manager(
#             {"waveform": waveform, "sample_rate": sample_rate},
#             num_speakers=num_speakers,
#             min_speakers=min_speakers or config.diarization.min_speakers,
#             max_speakers=max_speakers or config.diarization.max_speakers,
#         )
        
        
#         # Convert diarization output to the expected format
#         segments = []
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             segments.append({
#                 "speaker": speaker,
#                 "start": float(turn.start),
#                 "end": float(turn.end)
#             })
#         return DiarizationResponse(
#             diarization_segments=segments,
#             success=True
#         )
            
#     except Exception as e:
#         logger.exception("Diarization failed")
#         return DiarizationResponse(
#             diarization_segments=[],
#             success=False,
#             error=str(e)
#         )
        
# @router.post(
#     '/diarize', 
#     response_model=DiarizationResponse,
#     summary="Точечная диаризация аудиофайла")
# async def diarize(
#     config: ConfigDependency,
#     diarization_manager: any,
#     audio: UploadFile = File(...),
#     num_speakers: int | None = None,
#     min_speakers: int | None = None,
#     max_speakers: int | None = None
# ):
#     """
#     # Description
#         Эндпоинт для базовой диаризации аудио. Определяет говорящих в аудио файле.

#     #### Features:
#         - Определение количества говорящих
#         - Временные метки для каждого говорящего
#         - Настраиваемые параметры диаризации
        
#     ## Args:
#         config: Конфигурация приложения
#         audio: Аудио файл
#         num_speakers: Точное количество говорящих
#         min_speakers: Минимальное количество говорящих
#         max_speakers: Максимальное количество говорящих

#     ## Returns:
#         DiarizationResponse с результатами диаризации

#     #### Examples:
#         >>> with open('audio.wav', 'rb') as f:
#         ...     file = UploadFile(f)
#         ...     response = await diarize(
#         ...         config=config,
#         ...         audio=file,
#         ...         num_speakers=2
#         ...     )
#         >>> print(response.diarization_segments)
#         [{'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 1.5}, ...]
#     """
#     print(f'DIARIZATION MANAGER: {diarization_manager}')
#     return await diarize_audio(
#         config, diarization_manager, audio, num_speakers, min_speakers, max_speakers
#     )


@router.post(
    "/v1/audio/diarization",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
    summary="Диаризация аудиофайла (OpenAI формат)")
def diarize_file(
    config: ConfigDependency,
    model_manager: ModelManagerDependency,
    request: Request,
    diarization_audio: DiarizationAudioFileDependency,
    model: Annotated[ModelName | None, Form()] = None,
    language: Annotated[Language | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat | None, Form()] = None,
    temperature: Annotated[float, Form()] = 0.5,
    timestamp_granularities: Annotated[
        TimestampGranularities,
        # WARN: `alias` doesn't actually work.
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    # stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    vad_filter: Annotated[bool, Form()] = True,
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
    audio = diarization_audio['audio']
    filename = diarization_audio['filename']
    logger.info(f'Текущая модель: {model}')
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
        transcription_segments = list(TranscriptionSegment.from_faster_whisper_segments(segments))

    params = {'num_speakers': num_speakers} if num_speakers else {}
    
    logger.info(f'Transcription segments: {list(transcription_segments)}')
    
    # Convert numpy array to bytes
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
        
    # WAV формат очень важен для NeMo. Без него будет отрабатывать очень долго!
    filename = ''.join(filename.split('.')[:-1]) + '.wav'
    torchaudio.save(
        os.path.join('tmp', filename),
        torch.from_numpy(audio).cpu().unsqueeze(0),  # Add channel dimension
        sample_rate=16000,  # faster-whisper uses 16kHz
        channels_first=True,
        format='wav'
    )
    original_file_hash = get_file_hash('tmp/' + filename)
    logger.info(f'Успешно создали файл: tmp/{filename}')
    
    nemo_config, output_dir = create_basic_config(
        audio_filepath=filename, 
        num_speakers=num_speakers, 
        model_config_filepath='./diar_infer_telephonic.yaml'
    )
    
    # Send request with proper file formatting
    rttm_filename = ''.join(filename.split('.')[:-1]) + '.rttm'
    try:
        diarization_segments = diarize_nemo(nemo_config, output_dir + f'/pred_rttms/{rttm_filename}', "cuda", original_file_hash)
    except Exception as e:
        logger.error(f'Ошибка при диаризации: {e}')
        raise HTTPException(
            status_code=500,
            detail=f"Diarization failed: {e}"
        )
    else:
        logger.info(f'Успешно диаризовали файл!')
    
    # Map speakers to segments and return JSON response directly
    result = map_speakers_to_segments(transcription_segments, diarization_segments)
    logger.info(f'Результат диаризации: {result}')
    return Response(
            content=result,  # result is already a JSON string from map_speakers_to_segments
            media_type="application/json"
        )