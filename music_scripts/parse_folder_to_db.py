from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
import os
from consts import AUDIOS_DIR

import psycopg
load_dotenv()

target_dir = sys.argv[1] if len(sys.argv) > 1 else AUDIOS_DIR


def get_all_music_dirs():
    lst = []
    for f in Path(target_dir).iterdir():
        lst.append(f.name)
    return lst


def scan_game(game):
    t_dir = Path(target_dir) / game
    lst = []
    for f in Path(t_dir).rglob("*"):
        if f.is_dir():
            continue
        lst.append(f.relative_to(t_dir))
    return lst


def add_game_to_db(game):
    conn = psycopg.connect(os.environ["pghost"])
    cur = conn.cursor()
    musics = scan_game(game)
    for music in musics:
        cur.execute(
            "insert into music_data (game, path) values (%s, %s)", (str(game), str(music)))
    conn.commit()
    conn.close()


def sync_games_db():
    games = get_all_music_dirs()
    conn = psycopg.connect(os.environ["pghost"])
    cur = conn.cursor()
    for game in games:
        cur.execute(
            "select count(*) from music_data where game = %s limit 1", (str(game),))
        ans = cur.fetchone()
        if ans is not None and ans[0] != 0:
            print(f"{game} has already {ans} records")
        else:
            add_game_to_db(game)
    conn.commit()
    conn.close()


def sync_columns():
    conn = psycopg.connect(os.environ["pghost"])
    cur = conn.cursor()
    cur.execute("""
        update text_musics_rel set music_path = ltrim(music_path, '/') where music_path LIKE '/%' and music is null
    """)
    cur.execute(
        """
        update text_musics_rel t
            set music_path = m.path from music_data m
            where
                t.music is null and
                t.game = m.game and
                (trim(t.music_path) like 'UNKNOWN:%' or t.music_path is null) and
                (
                    substring(t.raw_music_line from 'audio\\.([A-Za-z0-9_]+)') = regexp_replace(m.path, '\\.([A-Za-z0-9]+)$', '') or        --- audio.filenamenoextension
                    lower(substring(trim(t.music_path) from 'UNKNOWN:(.+)$')) = lower(regexp_replace(m.path, '^.*/', '')) or                --- wrong path or only filename correct
                    lower(substring(trim(t.music_path) from 'UNKNOWN:(.+)$')) = lower(m.path)                                               --- path is used as variable
                )
        """
    )
    cur.execute(
        """
        update text_musics_rel t
            set music = m.id from music_data m
            where t.music is null and t.game = m.game and (
                t.music_path = m.path or                                                                                                        --- path is good
                m.path like '%' || t.music_path || '%' or                                                                                       --- path inclued in taken path
                regexp_replace(t.music_path, '<[^<>]+>','','g') = m.path or                                                                     --- path has comments
                regexp_replace(t.music_path, '\\.(%s|[a-zA-Z0-9]+)$', '') = regexp_replace(m.path, '\\.([a-zA-Z0-9]+)$', '') or                 --- path is accurate to the extension
                lower(regexp_replace(t.music_path, '\\.(%s|[a-zA-Z0-9]+)$', '')) = lower(regexp_replace(m.path, '\\.([a-zA-Z0-9]+)$', ''))      --- path is accurate to extension and case
            )
        """
    )
    conn.commit()
    conn.close()


def get_confused_cols():
    conn = psycopg.connect(os.environ["pghost"])
    cur = conn.cursor()
    cur.execute(
        "select game, music_path, sfx_before_path, raw_music_line, music from text_musics_rel where music is null and music_path != 'NO_MUSIC'")
    while True:
        temp = cur.fetchmany(10)
        if temp is None:
            break
        temp1 = input(temp)
        if temp1 != "":
            break
    conn.close()


if __name__ == "__main__":
    # get_confused_cols()
    # sync_games_db()
    sync_columns()
