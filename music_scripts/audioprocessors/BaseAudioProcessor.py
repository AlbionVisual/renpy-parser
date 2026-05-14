from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from psycopg import sql

from .norm_schema import NORM_METRICS_TABLE, ensure_norm_metrics_table, validate_pg_identifier

NormKind = Literal["none", "zscore"]
PreNormTransform = Literal["none", "log1p"]


@dataclass(frozen=True, slots=True)
class ColumnNormalizationSpec:
    column: str
    kind: NormKind = "zscore"
    eps: float = 1e-8
    pre_transform: PreNormTransform = "none"


class DBColumnDes:
    def __init__(self, name: str, t: str):
        self.name = name
        self.type = t

    def __repr__(self):
        return f"{self.name}: {self.type}"

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        return self.name == other.name


class BaseAudioProcessor:
    @property
    def processor_key(self) -> str:
        return self.__class__.__name__

    @property
    def target_columns(self) -> list[DBColumnDes]:
        raise NotImplementedError

    def process(self, y: np.ndarray, sr: int, path: str, ctx: dict) -> dict:
        raise NotImplementedError

    def column_normalization_specs(self) -> tuple[ColumnNormalizationSpec, ...]:
        """Как нормализовать числовые колонки процессора в БД (офлайн, по всей таблице).

        По умолчанию пусто — колонки этого процессора не входят в нормализующий апдейт.
        Перекройте метод и верните спецификации; ``normalize_db_col`` / ``normalize_samples``:
        1) по колонке считают ``avg`` и ``stddev_pop`` (опционально после ``ln(1+x)`` при ``pre_transform=log1p``);
        2) пишут тот же z-score с тем же преобразованием до вычитания среднего.

        Колонки, для которых нормализация бессмысленна (*duration_sec*, индексы и т.д.),
        намеренно не включать в результат.
        """
        return ()

    @staticmethod
    def zscore_specs_for_names(*column_names: str) -> tuple[ColumnNormalizationSpec, ...]:
        return tuple(ColumnNormalizationSpec(column=n) for n in column_names)

    def zscore_specs_for_target_columns_except(
        self,
        *excluded_column_names: str,
        log1p_for: frozenset[str] | None = None,
    ) -> tuple[ColumnNormalizationSpec, ...]:
        ex = frozenset(excluded_column_names)
        lp = log1p_for or frozenset()
        return tuple(
            ColumnNormalizationSpec(
                column=c.name,
                pre_transform="log1p" if c.name in lp else "none",
            )
            for c in self.target_columns
            if c.name not in ex
        )

    def normalize_db_col(self, cur, *, table_name: str) -> None:
        """Обучение: по текущей таблице считает mean/std по колонкам, пишет z-score обратно в БД и сохраняет метрики."""
        validate_pg_identifier(table_name)
        ensure_norm_metrics_table(cur)
        specs = [s for s in self.column_normalization_specs() if s.kind == "zscore"]
        if not specs:
            return
        pk = self.processor_key
        for spec in specs:
            col = validate_pg_identifier(spec.column)
            cid = sql.Identifier(col)
            tid = sql.Identifier(table_name)
            if spec.pre_transform == "log1p":
                xform = sql.SQL(
                    "ln(1.0::float8 + greatest({c}::float8, 0.0::float8))"
                ).format(c=cid)
            else:
                xform = sql.SQL("{}").format(cid)
            agg = sql.SQL(
                "select avg({xf})::float8, stddev_pop({xf})::float8 from {tbl} where {c} is not null"
            ).format(xf=xform, tbl=tid, c=cid)
            cur.execute(agg)
            row = cur.fetchone()
            if not row or row[0] is None:
                continue
            mean_v = float(row[0])
            raw_std = row[1]
            eps = float(spec.eps)
            if raw_std is None:
                denom = eps
            else:
                rf = float(raw_std)
                if rf != rf or rf < eps:
                    denom = eps
                else:
                    denom = rf
            upd = sql.SQL(
                "update {tbl} set {c} = ({xf} - %s::float8) / %s::float8 where {c} is not null"
            ).format(tbl=tid, c=cid, xf=xform)
            cur.execute(upd, (mean_v, denom))
            mt = sql.Identifier(NORM_METRICS_TABLE)
            ins = sql.SQL(
                """
                insert into {mt} (target_table, processor_key, column_name, mean, stddev)
                values (%s, %s, %s, %s, %s)
                on conflict (target_table, processor_key, column_name)
                do update set mean = excluded.mean, stddev = excluded.stddev
                """
            ).format(mt=mt)
            cur.execute(
                ins,
                (table_name, pk, col, mean_v, denom),
            )

    def normalize_samples(
        self,
        res: dict,
        *,
        metrics_by_column: dict[str, tuple[float, float]],
    ) -> dict:
        """Продакшен: те же коэффициенты что после ``normalize_db_col`` (снимок из БД или кэша)."""
        if not res or not metrics_by_column:
            return res
        out = dict(res)
        for spec in self.column_normalization_specs():
            if spec.kind != "zscore":
                continue
            col = spec.column
            if col not in out or col not in metrics_by_column:
                continue
            m, s = metrics_by_column[col]
            denom = max(float(s), float(spec.eps))
            v = float(out[col])
            if spec.pre_transform == "log1p":
                v = float(np.log1p(max(0.0, v)))
            out[col] = (v - float(m)) / denom
        return out
