import os
import psycopg
from consts import (
    COPY_CSV_TO_TEMP_TABLE_SQL,
    COPY_DB_TO_CSV_SQL,
    CREATE_TABLE_SQL,
    FIELDNAMES,
    FIELDNAMESINORDER,
    MUSIC_TABLE,
    TABLE_NAME,
    TEMP_TABLE_NAME,
)
from dotenv import load_dotenv

load_dotenv()


def createTables():
    conn = psycopg.connect(os.environ["pghost"])
    cur = conn.cursor()
    cur.execute(MUSIC_TABLE)
    cur.execute(CREATE_TABLE_SQL(TABLE_NAME))
    cur.execute(CREATE_TABLE_SQL(TEMP_TABLE_NAME))
    conn.commit()
    conn.close()


def ensureTables():
    conn = psycopg.connect(os.environ["pghost"])
    cur = conn.cursor()
    cur.execute(
        "select 1 from information_schema.tables where table_schema='public' and table_name= '"
        + TABLE_NAME
        + "'"
    )
    res = cur.fetchone() is not None
    cur.execute(
        "select 1 from information_schema.tables where table_schema='public' and table_name= '"
        + TEMP_TABLE_NAME
        + "'"
    )
    if cur.fetchone() is None:
        cur.execute(CREATE_TABLE_SQL(TEMP_TABLE_NAME))
        print("recreating temp table")
        conn.commit()
    conn.close()
    if not res:
        print("Table is not created, recreating...")
        createTables()


def pullCsv(filename):
    ensureTables()
    conn = psycopg.connect(os.environ["pghost"])
    cur = conn.cursor()
    with cur.copy(COPY_CSV_TO_TEMP_TABLE_SQL) as copy:
        with open(filename, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                copy.write(chunk)
    conn.commit()
    cols = [key for key in FIELDNAMES.values()]
    col_names = ", ".join(cols)
    set_clause = ", ".join(
        [
            c + " = EXCLUDED." + c
            for c in cols
            if c
            not in [
                FIELDNAMES["game name"],
                FIELDNAMES["file containing text"],
                FIELDNAMES["line in file"],
            ]
        ]
    )
    sql_request = (
        "insert into "
        + TABLE_NAME
        + "("
        + col_names
        + ") select "
        + col_names
        + " from "
        + TEMP_TABLE_NAME
        + " on conflict(game, file, line_number) do update set "
        + set_clause
    )
    cur.execute(sql_request)
    cur.execute("delete from " + TEMP_TABLE_NAME)
    conn.commit()
    conn.close()


def pushCsv(filename, bytes_lim=None, free_db=False):
    if os.path.exists(filename):
        raise FileExistsError("File '" + filename + "' already exists.")
    conn = psycopg.connect(os.environ["pghost"])
    cur = conn.cursor()
    cur.execute("select game from " + TABLE_NAME + " group by game")
    games = [game[0] for game in cur.fetchall()]
    print(games)
    written = 0
    with open(filename, "wb") as f:
        for game in games:
            with cur.copy(COPY_DB_TO_CSV_SQL("game = %s", written == 0), (game,)) as copy:
                for chunk in copy:
                    f.write(chunk)
                    written += len(chunk)
            if free_db:
                cur.execute("delete from " + TABLE_NAME + " where game= %s", (game,))
            if bytes_lim is not None and written > bytes_lim:
                break
    conn.commit()
    conn.close()


def insertDataset(dataset):
    ensureTables()
    conn = psycopg.connect(os.environ["pghost"])
    cur = conn.cursor()

    cols = list(dataset[0].keys())
    col_names = ", ".join(cols)
    sql = "copy " + TABLE_NAME + " (" + col_names + ") from stdin"

    with cur.copy(sql) as copy:
        for row in dataset:
            copy.write_row([row[col] for col in cols])

    conn.commit()
    conn.close()


if __name__ == "__main__":
    pushCsv("test.csv", 10 * 1000 * 1000)
