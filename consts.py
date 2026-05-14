from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_GAMES = Path(f"{BASE_DIR}/raw-games")
DOWNLOADS_DEFAULT = "/mnt/c/Users/Huawei/Downloads"
WORK_DIR = Path(f"{BASE_DIR}/.unpack_work")
OUTPUT_DIR = Path(f"{BASE_DIR}/output")
AUDIOS_DIR = Path(f"{BASE_DIR}/audios")
UNRPYC_PATH = Path(f"{BASE_DIR}/tools/unrpyc/unrpyc.py")
SKIP_FILES = {"screens.rpy", "gui.rpy", "options.rpy"}
STMT_KEYWORDS = {
    "scene", "show", "hide", "with", "play", "stop", "queue",
    "define", "default", "image", "transform", "style", "screen",
    "init", "python", "call", "jump", "return", "menu", "if",
    "elif", "else", "while", "for", "pass", "label", "translate",
    "window", "pause", "nvl", "at", "behind", "onlayer", "zorder",
    "camera", "$", "has", "centered",
    "add", "text", "button", "textbutton", "imagebutton",
    "hbox", "vbox", "frame", "grid", "fixed", "viewport",
    "use", "key", "timer", "on", "drag", "bar", "null",
    "input", "side", "action", "spacing", "xalign", "yalign",
    "xpos", "ypos", "xysize", "size", "idle_background",
    "hover_background", "selected_idle_background",
    "selected_hover_background", "tag",
}
BASE_INDENT = 4
TABLE_NAME = "text_musics_rel"
TEMP_TABLE_NAME = "text_musics_rel_temp"

MUSIC_TABLE = """
create table if not exists music_data(
    id serial primary key,
    game text not null,
    path text not null
)
"""


def CREATE_TABLE_SQL(t_name): return f"""
create table if not exists {t_name} (
    game text not null,
    file text not null,
    line_number int not null,
    phrase_order int,
    speaker text,
    text_orig text,
    text_translated text,
    text1 text,
    music_path text,
    music_looped boolean,
    raw_music_line text,
    sfx_before_path text,
    label text,
    extra jsonb,
    music integer references music_data(id),
    primary key (game, file, line_number)
)
"""


FIELDNAMES = {
    "order in text": "phrase_order",
    "who is speaking": "speaker",
    "original text": "text_orig",
    "text from translations": "text_translated",
    "path to bg music": "music_path",
    "is music looped": "music_looped",
    "line with music start": "raw_music_line",
    "path to sound effects": "sfx_before_path",
    "label of separable game parts": "label",
    "file containing text": "file",
    "line in file": "line_number",
    "game name": "game"
}
FIELDSORDER = [
    "game name",
    "original text",
    "text from translations",
    "path to bg music",
    "path to sound effects",
    "order in text",
    "is music looped",
    "who is speaking",
    "line with music start",
    "label of separable game parts",
    "file containing text",
    "line in file"
]
FIELDNAMESINORDER = [FIELDNAMES[key] for key in FIELDSORDER]
JOINER = ", "

COPY_CSV_TO_TEMP_TABLE_SQL = f"""
copy {TEMP_TABLE_NAME} ({JOINER.join(FIELDNAMESINORDER)})
from stdin
with (format csv, header true)
"""


def COPY_DB_TO_CSV_SQL(where_clause, is_start=False): return f"""
copy (select {JOINER.join(FIELDNAMESINORDER)} from text_musics_rel where {where_clause})
to stdout
with (format csv, header {is_start})
"""


if __name__ == "__main__":
    print(CREATE_TABLE_SQL(TEMP_TABLE_NAME))
    print(COPY_CSV_TO_TEMP_TABLE_SQL)
    print(COPY_DB_TO_CSV_SQL)
