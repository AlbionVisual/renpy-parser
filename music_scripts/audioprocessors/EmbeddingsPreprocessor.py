from .BaseAudioProcessor import BaseAudioProcessor, DBColumnDes
import numpy as np
from essentia.standard import TensorflowPredictEffnetDiscogs, MonoLoader

_MIN_16K_SAMPLES = max(8000, int(16000 * 0.5))


class EmbeddingsPreprocessor(BaseAudioProcessor):
    def __init__(self):
        self.model = TensorflowPredictEffnetDiscogs(
            graphFilename="music_scripts/audioprocessors/models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
        self._loader = MonoLoader()

    @property
    def target_columns(self) -> list[DBColumnDes]:
        return []

    def process(self, y: np.ndarray, sr: int, path: str, ctx: dict) -> dict:
        self._loader.configure(
            filename=ctx["full_path"],
            sampleRate=16000,
            resampleQuality=4,
        )
        audio = self._loader()
        if audio is None or len(audio) < _MIN_16K_SAMPLES:
            ctx["discogs_embeddings"] = None
            return {}
        ctx["discogs_embeddings"] = self.model(audio)
        return {}
