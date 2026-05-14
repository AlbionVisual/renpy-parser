from .BaseAudioProcessor import BaseAudioProcessor, DBColumnDes
import numpy as np
import librosa
from .music_consts import N_MELS, N_MFCC, HOP_LENGTH, N_FFT


class MFCCProcessor(BaseAudioProcessor):
    @property
    def target_columns(self) -> list[DBColumnDes]:
        return [
            DBColumnDes(f"mfcc_mean_{i:02d}", "double precision") for i in range(N_MFCC)
        ] + [
            DBColumnDes(f"mfcc_std_{i:02d}", "double precision") for i in range(N_MFCC)
        ] + [
            DBColumnDes(f"mfcc_p05_{i:02d}", "double precision") for i in range(N_MFCC)
        ] + [
            DBColumnDes(f"mfcc_p95_{i:02d}", "double precision") for i in range(N_MFCC)
        ]

    def process(self, y: np.ndarray, sr: int, path: str, ctx: dict) -> dict:
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
        S_db = librosa.power_to_db(S)
        mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_p05 = np.percentile(mfcc, 5, axis=1)
        mfcc_p95 = np.percentile(mfcc, 95, axis=1)
        out = {}
        for i in range(N_MFCC):
            out[f"mfcc_mean_{i:02d}"] = float(mfcc_mean[i])
            out[f"mfcc_std_{i:02d}"] = float(mfcc_std[i])
            out[f"mfcc_p05_{i:02d}"] = float(mfcc_p05[i])
            out[f"mfcc_p95_{i:02d}"] = float(mfcc_p95[i])
        return out

    def column_normalization_specs(self):
        return self.zscore_specs_for_target_columns_except()
