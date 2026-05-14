import os
import sys
from pathlib import Path

_music_scripts = Path(__file__).resolve().parent
_repo_root = _music_scripts.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_music_scripts))

import project_env

project_env.ensure_dotenv_loaded()

from audioprocessors.music_consts import TARGET_SR, MONO, MIN_WAVEFORM_SAMPLES
from audioprocessors.norm_schema import ensure_norm_metrics_table
from consts import AUDIOS_DIR
from audioprocessors.BaseAudioProcessor import BaseAudioProcessor, DBColumnDes
from audioprocessors.BasicStatsProcessor import BasicStatsProcessor
from audioprocessors.RythmProcessor import RythmProcessor
from audioprocessors.SpectralProcessor import SpectralProcessor
from audioprocessors.MFCCProcessor import MFCCProcessor
from audioprocessors.ChromaProcessor import ChromaProcessor
from audioprocessors.ClimaxProcessor import ClimaxProcessor
from audioprocessors.EmbeddingsPreprocessor import EmbeddingsPreprocessor
from audioprocessors.Predict2DProcessor import Predict2DProcessor
import librosa
import numpy as np
import psycopg
import re
from pprint import pprint
from random import shuffle

test_print = False


class AudioEnricher:
    def __init__(
        self,
        processors: list[BaseAudioProcessor],
        table_name: str = "music_data",
        *,
        apply_feature_normalization: bool = False,
    ):
        self.processors = processors
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name or ""):
            raise ValueError("Bad table_name: " + repr(table_name))
        self.table_name = table_name
        self.conn = psycopg.connect(os.environ["pghost"])
        self.cur = self.conn.cursor()
        self.apply_feature_normalization = apply_feature_normalization
        self._norm_metrics_cache: dict[str,
                                       dict[str, tuple[float, float]]] = {}
        self._norm_cache_loaded = False
        if apply_feature_normalization:
            self.refresh_norm_metrics_cache()

    def enrich_file(self, game, path) -> bool:
        if not self.update_table():
            return False
        full_path = AUDIOS_DIR / game / path
        y, sr = librosa.load(full_path, sr=TARGET_SR, mono=MONO)
        if y is None or len(y) < MIN_WAVEFORM_SAMPLES:
            print(
                "Skipped too-short audio",
                game,
                path,
                "samples",
                0 if y is None else len(y),
                "min",
                MIN_WAVEFORM_SAMPLES,
            )
            return False
        setters = ""
        values = []
        ctx = {"full_path": str(full_path)}
        for processor in self.processors:
            res = processor.process(y, sr, full_path, ctx)
            if self.apply_feature_normalization:
                if not self._norm_cache_loaded:
                    self.refresh_norm_metrics_cache()
                mk = self._norm_metrics_cache.get(processor.processor_key, {})
                res = processor.normalize_samples(res, metrics_by_column=mk)
            keys = list(res.keys())
            if len(setters) > 0 and len(keys) > 0:
                setters += ', '
            setters += ', '.join(
                [f"{col} = %s" for col in keys])
            values += [res[col] for col in keys]
        if setters == "" or len(values) == 0:
            return False
        values.append(game)
        values.append(path)
        sql = f"update {self.table_name} set {setters} where game = %s and path = %s"
        if test_print:
            print("AudioEnricher.enrich_file(38, sql): ", sql)
        self._query(sql, values, False)
        self._commit()
        return True

    def _get_must_have_cols(self) -> set[DBColumnDes]:
        must_have_cols = set()
        for proc in self.processors:
            for el in proc.target_columns:
                if el in must_have_cols:
                    print(
                        "AudioEnricher._get_must_have_cols(59): error, same column names found " + el.__repr__())
                    raise ValueError(
                        f"Same column names found: {el.__repr__()}")
                must_have_cols.add(el)
        return must_have_cols

    def update_table(self) -> bool:
        must_have_cols = self._get_must_have_cols()
        data = self._query(f"SELECT * FROM {self.table_name} LIMIT 0")
        existing_cols = set()
        for col in self._data():
            type_code = col.type_code
            data = self._query("select format_type(%s, null)", (type_code, ))
            el = DBColumnDes(col.name, data[0][0])
            existing_cols.add(el)
        if test_print:
            print("AudioEnricher.update_table(59, existing_cols): ", existing_cols)
        if test_print:
            print("AudioEnricher.update_table(60, must_have_cols): ", must_have_cols)
        to_add_cols = must_have_cols - existing_cols
        for col in to_add_cols:
            self._query(
                f"alter table {self.table_name} add column {col.name} {col.type}", do_return=False)
        if test_print:
            print("AudioEnricher.update_table(65, to_add_cols): ", to_add_cols)
        to_check_cols = must_have_cols & existing_cols
        existing_cols = list(existing_cols)
        must_have_cols = list(must_have_cols)
        for col in to_check_cols:
            for el in existing_cols:
                if el == col:
                    if el.type != col.type:
                        print(
                            "AudioEnricher.update_table: column type mismatch for",
                            repr(el.name),
                            "- in DB:",
                            repr(el.type),
                            "- expected by processor:",
                            repr(col.type),
                            "- no schema change was applied.",
                        )
                        if el.type == "integer" and col.type == "double precision":
                            print(
                                "Suggested fix:",
                                "alter table",
                                self.table_name,
                                "alter column",
                                col.name,
                                "type double precision using",
                                col.name + "::double precision;",
                            )
                        self._rollback()
                        return False
        if test_print:
            print("AudioEnricher.update_table(78, to_check_cols): ", to_check_cols)
        self._commit()
        return True

    def get_all_unworked_paths(self) -> list[tuple[str, str]]:
        return self._get_all_paths(False)

    def _get_all_paths(self, is_worked: bool = False, include_data: bool = False) -> list[tuple]:
        selector = "game, path" if not include_data else "*"
        suffix = " is not null" if is_worked else " is null"
        connector = " and " if is_worked else " or "
        must_have_cols = self._get_must_have_cols()
        cols = [col.name + suffix for col in must_have_cols]
        where_clause = connector.join(cols)
        if len(cols) == 0:
            where_clause = "True"
        res = self._query(
            f"select {selector} from {self.table_name} where {where_clause}")
        return res

    def _query(self, sql: str, params: tuple = None, do_return: bool = True) -> list[tuple]:
        try:
            self.cur.execute(sql, params)
            if do_return:
                return self.cur.fetchall()
        except (psycopg.OperationalError, psycopg.InterfaceError) as e:
            if self.conn.closed or self.cur.closed:
                print(
                    "AudioEnricher._query(115): error, connection or cursor was closed. Reconnecting and retrying...")
                self.conn = psycopg.connect(os.environ["pghost"])
                self.cur = self.conn.cursor()
                try:
                    self.cur.execute(sql, params)
                    if do_return:
                        return self.cur.fetchall()
                except Exception as e2:
                    print(
                        f"AudioEnricher._query(122): error, failed again after reconnect: {e2}")
                    self.conn.rollback()
                    raise
            else:
                print(f"AudioEnricher._query(126): error, database error: {e}")
                self.conn.rollback()
                raise

    def _commit(self):
        self.conn.commit()

    def _rollback(self):
        self.conn.rollback()

    def _data(self):
        return self.cur.description

    def get_worked_paths(self) -> list[tuple[str, str]]:
        return self._get_all_paths(True)

    def get_worked_data(self) -> list[tuple]:
        return self._get_all_paths(True, True)

    def refresh_norm_metrics_cache(self) -> None:
        """Прод: перечитать mean/std из ``audio_processor_norm_metrics`` под текущий ``table_name``."""
        ensure_norm_metrics_table(self.cur)
        self._norm_metrics_cache.clear()
        for p in self.processors:
            specs = p.column_normalization_specs()
            if not specs:
                continue
            pk = p.processor_key
            self.cur.execute(
                """
                select column_name, mean, stddev
                from audio_processor_norm_metrics
                where target_table = %s and processor_key = %s
                """,
                (self.table_name, pk),
            )
            self._norm_metrics_cache[pk] = {
                r[0]: (float(r[1]), float(r[2]))
                for r in self.cur.fetchall()
                if r[1] is not None and r[2] is not None
            }
        self._norm_cache_loaded = True

    def finalize_training_normalization(self) -> None:
        """После заполнения сырыми фичами: z-score по колонке на всём ``table_name`` и upsert метрик."""
        ensure_norm_metrics_table(self.cur)
        for p in self.processors:
            p.normalize_db_col(self.cur, table_name=self.table_name)
        self.conn.commit()
        self._norm_cache_loaded = False


SMOKE_ENRICH: list[tuple[str, str]] = [
    # ("SmolBirb-pc", "audio/ambient4blues.mp3"),
]
RUN_RANDOM_UNWORKED: int = 20


if __name__ == "__main__":
    from time import time
    processors = [
        BasicStatsProcessor(), RythmProcessor(),
        SpectralProcessor(), MFCCProcessor(),
        ChromaProcessor(), ClimaxProcessor(),
        EmbeddingsPreprocessor(),
        Predict2DProcessor(
            "music_scripts/audioprocessors/models/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
            "music_scripts/audioprocessors/models/mtg_jamendo_moodtheme-discogs-effnet-1.json",
            "jamendo_moodtheme"),
        Predict2DProcessor(
            "music_scripts/audioprocessors/models/mtg_jamendo_genre-discogs-effnet-1.pb",
            "music_scripts/audioprocessors/models/mtg_jamendo_genre-discogs-effnet-1.json",
            "jamendo_genre"),
        Predict2DProcessor(
            "music_scripts/audioprocessors/models/mtg_jamendo_instrument-discogs-effnet-1.pb",
            "music_scripts/audioprocessors/models/mtg_jamendo_instrument-discogs-effnet-1.json",
            "jamendo_instrument"),
    ]
    print("processors loaded")
    enr = AudioEnricher(processors, "music_data")
    print("enricher loaded")
    for game, path in SMOKE_ENRICH:
        start_time = time()
        if enr.enrich_file(game, path):
            print("Enriched", game, path, "in", time() - start_time, "seconds")
        else:
            print("Skipped", game, path,
                  "- update_table failed or nothing to write")
    if RUN_RANDOM_UNWORKED > 0:
        unworked = enr.get_all_unworked_paths()
        shuffle(unworked)
        unworked = unworked[:RUN_RANDOM_UNWORKED]
        for game, path in unworked:
            try:
                start_time = time()
                if enr.enrich_file(game, path):
                    print("Enriched", game, "---", path, "in",
                          time() - start_time, "seconds")
                else:
                    print("Skipped", game, "---", path,
                          "- update_table failed or nothing to write")
            except Exception as e:
                print(str(e))
    print("\nNormalization start")
    enr.finalize_training_normalization()
    print("Normalization finished\n")
    worked = enr.get_worked_data()
    if worked:
        # pprint(worked[0])
        print(
            len(worked),
            "in total. Every row has",
            len(worked[0]),
            "columns",
        )
    else:
        print("No worked rows to print (get_worked_data empty).")
