import re

NORM_METRICS_TABLE = "audio_processor_norm_metrics"

CREATE_NORM_METRICS_SQL = """
create table if not exists audio_processor_norm_metrics (
    target_table text not null,
    processor_key text not null,
    column_name text not null,
    mean double precision not null,
    stddev double precision not null,
    primary key (target_table, processor_key, column_name)
);
"""


def ensure_norm_metrics_table(cur) -> None:
    cur.execute(CREATE_NORM_METRICS_SQL)


def validate_pg_identifier(name: str) -> str:
    if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("invalid sql identifier: " + repr(name))
    return name
