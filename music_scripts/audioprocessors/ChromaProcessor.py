from .BaseAudioProcessor import BaseAudioProcessor, DBColumnDes
import numpy as np
from .music_consts import HOP_LENGTH
import librosa


class ChromaProcessor(BaseAudioProcessor):

    def __init__(self, stft_instead_cqt: bool = False):
        self.stft_instead_cqt = stft_instead_cqt

    @property
    def target_columns(self) -> list[DBColumnDes]:
        return [
            DBColumnDes(f"chroma_mean_{i:02d}", "double precision") for i in range(12)
        ] + [
            DBColumnDes("chroma_entropy", "double precision"),
            DBColumnDes("chroma_peakiness", "double precision"),
        ]

    def process(self, y: np.ndarray, sr: int, path: str, ctx: dict) -> dict:
        func = librosa.feature.chroma_stft if self.stft_instead_cqt else librosa.feature.chroma_cqt
        chroma = func(y=y, sr=sr, hop_length=HOP_LENGTH)
        chroma_mean = np.mean(chroma, axis=1)
        p = chroma_mean / (np.sum(chroma_mean) + 1e-9)
        chroma_entropy = float(-np.sum(p * np.log(p + 1e-9)))
        chroma_peakiness = float(np.max(chroma_mean) /
                                 (np.mean(chroma_mean) + 1e-9))
        out = {}
        for i in range(12):
            out[f"chroma_mean_{i:02d}"] = float(chroma_mean[i])
        out["chroma_entropy"] = chroma_entropy
        out["chroma_peakiness"] = chroma_peakiness
        return out

    def column_normalization_specs(self):
        return self.zscore_specs_for_target_columns_except()
