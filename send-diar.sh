#!/bin/bash

# Advanced diarization request with whisper transcription
echo "\n\nTesting advanced diarization endpoint..."
curl -X POST "http://0.0.0.0:8000/v1/audio/diarization" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@audio.wav" \
        -F "language=ru" \
        -F "response_format=verbose_json" \
        -F "num_speakers=2" \
        -F "timestamp_granularities=segment" \
        -F "timestamp_granularities=word"