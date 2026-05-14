from .BaseAudioProcessor import BaseAudioProcessor, DBColumnDes
import numpy as np
import librosa
from .music_consts import HOP_LENGTH


class RythmProcessor(BaseAudioProcessor):
    @property
    def target_columns(self) -> list[DBColumnDes]:
        return [
            DBColumnDes("tempo", "double precision"),
            DBColumnDes("onset_rate_hz", "double precision"),
            DBColumnDes("beat_strength_mean", "double precision"),
            DBColumnDes("beat_strength_std", "double precision"),
            # DBColumnDes("tempo_confidence", "double precision"),
        ]

    def process(self, y: np.ndarray, sr: int, path: str, ctx: dict) -> dict:
        duration_sec = librosa.get_duration(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=HOP_LENGTH)
        tempo_bpm, _ = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
        if isinstance(tempo_bpm, np.ndarray):
            if len(tempo_bpm) == 1:
                tempo_bpm = float(tempo_bpm[0])
            elif len(tempo_bpm) > 1:
                tempo_bpm = float(np.median(tempo_bpm))
            else:
                raise ValueError(
                    f"Unexpected tempo_bpm length: {len(tempo_bpm)}")
        else:
            tempo_bpm = float(tempo_bpm)
        onset_times = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, units="time")
        onset_rate_hz = float(len(onset_times) / duration_sec)
        beat_strength_mean = float(onset_env.mean())
        beat_strength_std = float(onset_env.std())
        return {
            "tempo": tempo_bpm,
            "onset_rate_hz": onset_rate_hz,
            "beat_strength_mean": beat_strength_mean,
            "beat_strength_std": beat_strength_std,
        }

    def column_normalization_specs(self):
        return self.zscore_specs_for_target_columns_except()
