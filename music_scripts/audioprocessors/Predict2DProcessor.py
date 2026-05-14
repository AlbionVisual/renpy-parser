from .BaseAudioProcessor import BaseAudioProcessor, DBColumnDes
import numpy as np
from essentia.standard import TensorflowPredict2D
import json


class Predict2DProcessor(BaseAudioProcessor):
    def __init__(self, model_path: str, model_meta_path: str, name_in_db: str, combine_method=lambda arr: np.quantile(arr, 0.95, axis=0)):
        self.model = TensorflowPredict2D(
            graphFilename=model_path)
        meta = json.load(open(
            model_meta_path, "r", encoding="utf-8"))
        self.prefix = name_in_db + "_"
        self.classes = [self.prefix + cls for cls in meta["classes"]]
        self.combine_method = combine_method

    @property
    def processor_key(self) -> str:
        return self.prefix.rstrip("_")

    @property
    def target_columns(self) -> list[DBColumnDes]:
        return [DBColumnDes(col, "double precision") for col in self.classes]

    def process(self, y: np.ndarray, sr: int, path: str, ctx: dict) -> dict:
        emb = ctx.get("discogs_embeddings")
        if emb is None:
            return dict.fromkeys(self.classes, 0.0)
        try:
            raw = self.model(emb)
        except TypeError:
            return dict.fromkeys(self.classes, 0.0)
        arr = np.asarray(raw)
        if arr.size == 0:
            return dict.fromkeys(self.classes, 0.0)
        predictions = self.combine_method(arr).astype(float)
        res = dict(zip(self.classes, predictions))
        return res

    def column_normalization_specs(self):
        return ()
