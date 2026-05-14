from .BaseAudioProcessor import BaseAudioProcessor, DBColumnDes
import numpy as np
import librosa
from .music_consts import CLIMAX_Q, CLIMAX_MIN_DISTANCE_SEC, HOP_LENGTH


class ClimaxProcessor(BaseAudioProcessor):
    def __init__(self, rms_instead_onset: bool = False):
        self.rms_instead_onset = rms_instead_onset

    @property
    def target_columns(self) -> list[DBColumnDes]:
        return [
            DBColumnDes("climax_count", "double precision"),
            DBColumnDes("climax_density_per_min", "double precision"),
            DBColumnDes("climax_strength_mean", "double precision"),
            DBColumnDes("climax_strength_max", "double precision"),
            DBColumnDes("climax_strength_p95", "double precision"),
            DBColumnDes("energy_change_mean", "double precision"),
            DBColumnDes("energy_change_p95", "double precision"),
        ]

    def process(self, y: np.ndarray, sr: int, path: str, ctx: dict) -> dict:
        duration_sec = len(y) / sr
        if self.rms_instead_onset:
            series = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        else:
            series = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=HOP_LENGTH)
        novelty = np.maximum(0.0, np.diff(series, prepend=series[0]))
        thr = float(np.quantile(novelty, CLIMAX_Q))
        min_distance = int(CLIMAX_MIN_DISTANCE_SEC * sr / HOP_LENGTH)

        peak_idx = librosa.util.peak_pick(
            novelty,
            pre_max=1,
            post_max=1,
            pre_avg=1,
            post_avg=1,
            delta=0.0,
            wait=min_distance)
        peak_idx = peak_idx[novelty[peak_idx] >= thr]

        count = int(len(peak_idx))
        density_per_min = float(count / (duration_sec / 60.0))

        if count > 0:
            strengths = novelty[peak_idx]
            strength_mean = float(np.mean(strengths))
            strength_max = float(np.max(strengths))
            strength_p95 = float(np.quantile(strengths, 0.95))
        else:
            strength_mean = 0.0
            strength_max = 0.0
            strength_p95 = 0.0

        energy_change = np.abs(np.diff(series, prepend=series[0]))
        energy_change_mean = float(np.mean(energy_change))
        energy_change_p95 = float(np.quantile(energy_change, 0.95))

        return {
            "climax_count": float(count),
            "climax_density_per_min": density_per_min,
            "climax_strength_mean": strength_mean,
            "climax_strength_max": strength_max,
            "climax_strength_p95": strength_p95,
            "energy_change_mean": energy_change_mean,
            "energy_change_p95": energy_change_p95,
        }

    def column_normalization_specs(self):
        return self.zscore_specs_for_target_columns_except(
            log1p_for=frozenset({"climax_count"}),
        )
