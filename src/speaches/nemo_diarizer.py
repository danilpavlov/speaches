from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
import json
import os
from uuid import uuid4
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, read_rttm_lines
from speaches.map_speakers import DiarizationSegment
import logging
import shutil


logger = logging.getLogger(__name__)

def create_basic_config(audio_filepath: str, num_speakers: int, model_config_filepath: str):
    diarization_id = str(uuid4())
    diarization_dir = os.path.join('tmp', diarization_id)
    manifest_filepath = os.path.join(diarization_dir, 'input_manifest.json')
    if not os.path.exists(diarization_dir):
        os.makedirs(diarization_dir, exist_ok=True)
    meta = {
        'audio_filepath': f'./tmp/{audio_filepath}', #audio_filepath, 
        'offset': 0, 
        'duration':None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': num_speakers, 
        'rttm_filepath': None, 
        'uem_filepath' : None
    }
    
    with open(manifest_filepath,'w') as fp:
        json.dump(meta,fp)
        fp.write('\n')
        
    output_dir = os.path.join(diarization_dir, 'oracle_vad')
    os.makedirs(output_dir, exist_ok=True)
    
    config = OmegaConf.load(model_config_filepath)
    config.diarizer.manifest_filepath = manifest_filepath
    config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
    
    
    logger.info(f'Успешно создали конфиг: {config}')
    
    return config, output_dir


def diarize(config: dict, output_rttm_filepath: str, device: str, original_file_hash: str) -> list[DiarizationSegment]:
    # Starting diarization:
    
    os.makedirs('rttm_cache', exist_ok=True)
    if f'{original_file_hash}.rttm' in os.listdir('rttm_cache'):
        logger.info(f'Используем кэшированный rttm файл: {original_file_hash}.rttm')
        output_rttm_filepath = os.path.join('rttm_cache', f'{original_file_hash}.rttm')
        labels = rttm_to_labels(output_rttm_filepath)
    else:
        oracle_vad_clusdiar_model = ClusteringDiarizer(cfg=config).to(device)
        oracle_vad_clusdiar_model.diarize()
        rttm_lines = read_rttm_lines(output_rttm_filepath)
        print(rttm_lines)
        with open('rttm_cache/'+f'{original_file_hash}.rttm', 'w') as f:
            f.write(''.join(rttm_lines))
            labels = rttm_to_labels(output_rttm_filepath)
            
        shutil.rmtree('./tmp/')
    
    # Extracting segment labels:
    diar_segments = [] 
    for label in labels:
        start, end, speaker = label.split()
        diar_segments.append(DiarizationSegment(speaker=speaker, start=float(start), end=float(end))) 
    
    return diar_segments
    

    
    
    

