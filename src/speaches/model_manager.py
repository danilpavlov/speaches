"""
Модуль для управления загрузкой и выгрузкой моделей Whisper, Piper и Kokoro.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import gc
import json
import logging
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession

from speaches.hf_utils import get_kokoro_model_path, get_piper_voice_model_file

if TYPE_CHECKING:
    from collections.abc import Callable

    from piper.voice import PiperVoice

    from speaches.config import WhisperConfig

logger = logging.getLogger(__name__)

# TODO: enable concurrent model downloads


class SelfDisposingModel[T]:
    """
    Класс для автоматического управления временем жизни модели.
    """
    def __init__(
        self, 
        model_id: str, 
        load_fn: Callable[[], T], 
        ttl: int, 
        unload_fn: Callable[[str], None] | None = None,
        enable_dynamic_loading: bool = False
    ) -> None:
        self.model_id = model_id
        self.load_fn = load_fn
        self.ttl = ttl
        self.unload_fn = unload_fn

        self.ref_count: int = 0
        self.rlock = threading.RLock() if enable_dynamic_loading else nullcontext()
        self.expire_timer: threading.Timer | None = None
        self.model: T | None = None

    def unload(self) -> None:
        """
        Description
            Выгружает модель из памяти.
        """
        if isinstance(self.rlock, nullcontext):
            logger.info('Dynamic loading is disabled, skipping unload')
            return None
        with self.rlock:
            if self.model is None:
                raise ValueError(f"Model {self.model_id} is not loaded. {self.ref_count=}")
            if self.ref_count > 0:
                raise ValueError(f"Model {self.model_id} is still in use. {self.ref_count=}")
            if self.expire_timer:
                self.expire_timer.cancel()
            self.model = None
            # WARN: ~300 MB of memory will still be held by the model. See https://github.com/SYSTRAN/faster-whisper/issues/992
            gc.collect()
            logger.info(f"Model {self.model_id} unloaded")
            if self.unload_fn is not None:
                self.unload_fn(self.model_id)

    def _load(self) -> None:
        with self.rlock:
            assert self.model is None
            logger.debug(f"Loading model {self.model_id}")
            start = time.perf_counter()
            self.model = self.load_fn()
            logger.info(f"Model {self.model_id} loaded in {time.perf_counter() - start:.2f}s")

    def _increment_ref(self) -> None:
        with self.rlock:
            self.ref_count += 1
            if self.expire_timer:
                logger.debug(f"Model was set to expire in {self.expire_timer.interval}s, cancelling")
                self.expire_timer.cancel()
            logger.debug(f"Incremented ref count for {self.model_id}, {self.ref_count=}")

    def _decrement_ref(self) -> None:
        with self.rlock:
            self.ref_count -= 1
            logger.debug(f"Decremented ref count for {self.model_id}, {self.ref_count=}")
            if self.ref_count <= 0:
                if self.ttl > 0:
                    logger.info(f"Model {self.model_id} is idle, scheduling offload in {self.ttl}s")
                    self.expire_timer = threading.Timer(self.ttl, self.unload)
                    self.expire_timer.start()
                elif self.ttl == 0:
                    logger.info(f"Model {self.model_id} is idle, unloading immediately")
                    self.unload()
                else:
                    logger.info(f"Model {self.model_id} is idle, not unloading")

    def __enter__(self) -> T:
        with self.rlock:
            if self.model is None:
                self._load()
            self._increment_ref()
            assert self.model is not None
            return self.model

    def __exit__(self, *_args) -> None:  # noqa: ANN002
        self._decrement_ref()


class WhisperModelManager:
    """
    Класс для управления моделями Whisper.
    """
    def __init__(self, whisper_config: WhisperConfig, enable_dynamic_loading: bool = False) -> None:
        self.whisper_config = whisper_config
        self.loaded_models: OrderedDict[str, SelfDisposingModel[WhisperModel]] = OrderedDict()
        self._lock = threading.Lock() if enable_dynamic_loading else nullcontext()
        

    def _load_fn(self, model_id: str) -> WhisperModel:
        return WhisperModel(
            model_id,
            device=self.whisper_config.inference_device,
            device_index=self.whisper_config.device_index,
            compute_type=self.whisper_config.compute_type,
            cpu_threads=self.whisper_config.cpu_threads,
            num_workers=self.whisper_config.num_workers,
            local_files_only=True  # True, if you don't want to load from the internet and check cache only
        )

    def _handle_model_unload(self, model_name: str) -> None:
        with self._lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]

    def unload_model(self, model_name: str) -> None:
        """
        Description
            Выгружает модель Whisper из памяти.

        Args:
            model_name: Имя модели.
        """
        with self._lock:
            model = self.loaded_models.get(model_name)
            if model is None:
                raise KeyError(f"Model {model_name} not found")
            self.loaded_models[model_name].unload()

    def load_model(self, model_name: str) -> SelfDisposingModel[WhisperModel]:
        """
        Description
            Загружает модель Whisper в память.

        Args:
            model_name: Имя модели.

        Returns:
            Объект SelfDisposingModel с загруженной моделью Whisper.
        """
        with self._lock:
            logger.debug("Acquired lock")
            if model_name in self.loaded_models:
                logger.debug(f"{model_name} model already loaded")
                return self.loaded_models[model_name]
            self.loaded_models[model_name] = SelfDisposingModel[WhisperModel](
                model_name,
                load_fn=lambda: self._load_fn(model_name),
                ttl=self.whisper_config.ttl,
                unload_fn=self._handle_model_unload,
            )
            return self.loaded_models[model_name]


ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class PiperModelManager:
    """
    Класс для управления моделями Piper.
    """
    def __init__(self, ttl: int, enable_dynamic_loading: bool = False) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[PiperVoice]] = OrderedDict()
        self._lock = threading.Lock() if enable_dynamic_loading else nullcontext()

    def _load_fn(self, model_id: str) -> PiperVoice:
        from piper.voice import PiperConfig, PiperVoice

        model_path = get_piper_voice_model_file(model_id)
        inf_sess = InferenceSession(model_path, providers=ONNX_PROVIDERS)
        config_path = Path(str(model_path) + ".json")
        conf = PiperConfig.from_dict(json.loads(config_path.read_text()))
        return PiperVoice(session=inf_sess, config=conf)

    def _handle_model_unload(self, model_name: str) -> None:
        with self._lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]

    def unload_model(self, model_name: str) -> None:
        """
        Description
            Выгружает модель Piper из памяти.

        Args:
            model_name: Имя модели.
        """
        with self._lock:
            model = self.loaded_models.get(model_name)
            if model is None:
                raise KeyError(f"Model {model_name} not found")
            self.loaded_models[model_name].unload()

    def load_model(self, model_name: str) -> SelfDisposingModel[PiperVoice]:
        """
        Description
            Загружает модель Piper в память.

        Args:
            model_name: Имя модели.

        Returns:
            Объект SelfDisposingModel с загруженной моделью Piper.
        """
        from piper.voice import PiperVoice

        with self._lock:
            if model_name in self.loaded_models:
                logger.debug(f"{model_name} model already loaded")
                return self.loaded_models[model_name]
            self.loaded_models[model_name] = SelfDisposingModel[PiperVoice](
                model_name,
                load_fn=lambda: self._load_fn(model_name),
                ttl=self.ttl,
                unload_fn=self._handle_model_unload,
            )
            return self.loaded_models[model_name]


class KokoroModelManager:
    """
    Класс для управления моделями Kokoro.
    """
    def __init__(self, ttl: int, enable_dynamic_loading: bool = False) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[Kokoro]] = OrderedDict()
        self._lock = threading.Lock() if enable_dynamic_loading else nullcontext()

    def _load_fn(self, _model_id: str) -> Kokoro:
        model_path = get_kokoro_model_path()
        voices_path = model_path.parent / "voices.json"
        inf_sess = InferenceSession(model_path, providers=ONNX_PROVIDERS)
        return Kokoro.from_session(inf_sess, str(voices_path))

    def _handle_model_unload(self, model_name: str) -> None:
        with self._lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]

    def unload_model(self, model_name: str) -> None:
        """
        Description
            Выгружает модель Kokoro из памяти.

        Args:
            model_name: Имя модели.
        """
        with self._lock:
            model = self.loaded_models.get(model_name)
            if model is None:
                raise KeyError(f"Model {model_name} not found")
            self.loaded_models[model_name].unload()

    def load_model(self, model_name: str) -> SelfDisposingModel[Kokoro]:
        """
        Description
            Загружает модель Kokoro в память.

        Args:
            model_name: Имя модели.

        Returns:
            Объект SelfDisposingModel с загруженной моделью Kokoro.
        """
        with self._lock:
            if model_name in self.loaded_models:
                logger.debug(f"{model_name} model already loaded")
                return self.loaded_models[model_name]
            self.loaded_models[model_name] = SelfDisposingModel[Kokoro](
                model_name,
                load_fn=lambda: self._load_fn(model_name),
                ttl=self.ttl,
                unload_fn=self._handle_model_unload,
            )
            return self.loaded_models[model_name]