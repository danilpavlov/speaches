#!/bin/bash

echo "Отправка 5 параллельных запросов дианаризации..."

curl -X POST "http://0.0.0.0:8000/v1/audio/diarization" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@audio.wav" \
    -F "language=ru" \
    -F "response_format=verbose_json" \
    -F "num_speakers=2" \
    -F "timestamp_granularities=segment" \
    -F "timestamp_granularities=word" & curl -X POST "http://0.0.0.0:8000/v1/audio/diarization" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@audio.wav" \
    -F "language=ru" \
    -F "response_format=verbose_json" \
    -F "num_speakers=2" \
    -F "timestamp_granularities=segment" \
    -F "timestamp_granularities=word" & curl -X POST "http://0.0.0.0:8000/v1/audio/diarization" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@audio.wav" \
    -F "language=ru" \
    -F "response_format=verbose_json" \
    -F "num_speakers=2" \
    -F "timestamp_granularities=segment" \
    -F "timestamp_granularities=word" & curl -X POST "http://0.0.0.0:8000/v1/audio/diarization" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@audio.wav" \
    -F "language=ru" \
    -F "response_format=verbose_json" \
    -F "num_speakers=2" \
    -F "timestamp_granularities=segment" \
    -F "timestamp_granularities=word" 