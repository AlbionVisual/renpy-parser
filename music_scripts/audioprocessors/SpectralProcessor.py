from .BaseAudioProcessor import BaseAudioProcessor, DBColumnDes
import numpy as np
import librosa
from .music_consts import HOP_LENGTH, N_FFT


class SpectralProcessor(BaseAudioProcessor):
    @property
    def target_columns(self) -> list[DBColumnDes]:
        data1 = ["centroid", "bandwidth", "rolloff", "flatness", "zcr"]
        data2 = ["mean", "std", "p05", "p50", "p95"]
        return [DBColumnDes(f"spectral_{dt1}_{dt2}", "double precision") for dt1 in data1 for dt2 in data2]

    def process(self, y: np.ndarray, sr: int, path: str, ctx: dict) -> dict:
        S = np.abs(librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(
            S=S, sr=sr, roll_percent=0.85)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)[0]
        centroid_data = self._get_data(centroid, "spectral_centroid")
        bandwidth_data = self._get_data(bandwidth, "spectral_bandwidth")
        rolloff_data = self._get_data(rolloff, "spectral_rolloff")
        flatness_data = self._get_data(flatness, "spectral_flatness")
        zcr_data = self._get_data(zcr, "spectral_zcr")
        return {
            **centroid_data,
            **bandwidth_data,
            **rolloff_data,
            **flatness_data,
            **zcr_data,
        }

    def _get_data(self, arr: np.ndarray, prefix: str = "") -> dict:
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        p05 = float(np.percentile(arr, 5))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
            f"{prefix}_p05": p05,
            f"{prefix}_p50": p50,
            f"{prefix}_p95": p95,
        }

    def column_normalization_specs(self):
        return self.zscore_specs_for_target_columns_except()
