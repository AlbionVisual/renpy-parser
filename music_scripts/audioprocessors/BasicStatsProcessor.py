from .BaseAudioProcessor import BaseAudioProcessor, DBColumnDes
import numpy as np
import librosa
from .music_consts import HOP_LENGTH, RMS_FRAME_LENGTH


class BasicStatsProcessor(BaseAudioProcessor):
    @property
    def target_columns(self) -> list[DBColumnDes]:
        return [
            DBColumnDes("duration_sec", "double precision"),
            DBColumnDes("rms_mean", "double precision"),
            DBColumnDes("rms_std", "double precision"),
            DBColumnDes("rms_p05", "double precision"),
            DBColumnDes("rms_p95", "double precision"),
            DBColumnDes("rms_max", "double precision"),
            DBColumnDes("dyn_range_p95_p05", "double precision"),
        ]

    def process(self, y: np.ndarray, sr: int, path: str, ctx: dict) -> dict:
        duration_sec = float(librosa.get_duration(y=y, sr=sr))
        rms = librosa.feature.rms(
            y=y, hop_length=HOP_LENGTH, frame_length=RMS_FRAME_LENGTH)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        rms_p05 = float(np.percentile(rms, 5))
        rms_p95 = float(np.percentile(rms, 95))
        rms_max = float(np.max(rms))
        dyn_range_p95_p05 = rms_p95 - rms_p05
        return {
            "duration_sec": duration_sec,
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "rms_p05": rms_p05,
            "rms_p95": rms_p95,
            "rms_max": rms_max,
            "dyn_range_p95_p05": dyn_range_p95_p05,
        }

    def column_normalization_specs(self):
        return self.zscore_specs_for_target_columns_except("duration_sec")
