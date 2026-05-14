import re
from pathlib import Path
from consts import SKIP_FILES, STMT_KEYWORDS

RE_LABEL = re.compile(r"^\s*label\s+([\w.]+)\s*(?:\(.*\))?\s*:", re.MULTILINE)
RE_DEFINE = re.compile(
    r"^\s*(?:define|default)\s+(?:audio\.\s*)?(\w+)\s*=\s*[\"']([^\"']+)[\"']",
    re.MULTILINE,
)
RE_PY_VAR = re.compile(r"^\s*\$?\s*(\w+)\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE)
RE_DEFINE_VAR = re.compile(
    r"^\s*(?:define|default)\s+([\w.]+)\s*=\s*(.+)",
    re.MULTILINE,
)
RE_PY_DICT_BLOCK = re.compile(
    r"^\s*(\w+)\s*=\s*\{([^\}]*)\}\s*$",
    re.MULTILINE,
)
RE_DICT_STR_PAIR = re.compile(
    r'["\']([^"\']+)["\']\s*:\s*["\']((?:[^"\'\\]|\\.)*)["\']',
)
RE_MUSIC_LIST_BRACKET_ASSIGN = re.compile(
    r"^\s*\$?\s*music_list\[\s*[\"']([^\"']+)[\"']\s*\]\s*=\s*[\"']((?:[^\"'\\]|\\.)*)[\"']",
)
RE_TL_LINE = re.compile(
    r"^\s*translate\s+\w+\s+([\w.]+)\s*:\s*$", re.IGNORECASE
)

RU_TL_FOLDERS = ("russian", "rus", "ru")


def findRpyFiles(game_dir):
    game_dir = Path(game_dir)
    game_subdir = None
    for candidate in game_dir.rglob("game"):
        if candidate.is_dir():
            game_subdir = candidate
            break
    if game_subdir is None:
        game_subdir = game_dir

    files = []
    for f in game_subdir.rglob("*.rpy"):
        if "/tl/" not in f.as_posix() and f.name not in SKIP_FILES:
            files.append(f)

    return sorted(files, key=lambda f: f.name), game_subdir


def scanLabels(rpy_files):
    labels = {}
    for f in rpy_files:
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        for m in RE_LABEL.finditer(text):
            name = m.group(1)
            line_no = text[:m.start()].count("\n") + 1
            labels[name] = {"file": f, "line": line_no}
    return labels


def scanDefines(rpy_files):
    audio_map = {}
    all_vars = {}
    for f in rpy_files:
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        for m in RE_DEFINE.finditer(text):
            audio_map[m.group(1)] = m.group(2)

        for m in RE_PY_VAR.finditer(text):
            var_name = m.group(1)
            var_val = m.group(2)
            if "/" in var_val or "." in var_val:
                audio_map[var_name] = var_val

        for m in RE_DEFINE_VAR.finditer(text):
            var_name = m.group(1)
            var_val = m.group(2).strip()
            line_no = text[:m.start()].count("\n") + 1
            all_vars[var_name] = {
                "value": var_val,
                "file": str(f),
                "line": line_no,
            }
    return audio_map, all_vars


def scanMusicKeys(rpy_files):
    out = {}
    for f in rpy_files:
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        for line in text.splitlines():
            mla = RE_MUSIC_LIST_BRACKET_ASSIGN.match(line)
            if not mla:
                continue
            k, v = mla.group(1), mla.group(2)
            if not re.search(r"\.(ogg|opus|mp3|wav)", v, re.I):
                continue
            out['music_list["' + k + '"]'] = v
        for m in RE_PY_DICT_BLOCK.finditer(text):
            dname, body = m.group(1), m.group(2)
            pairs = list(RE_DICT_STR_PAIR.finditer(body))
            if not pairs:
                continue
            path_like = False
            for pm in pairs:
                if re.search(r"\.(ogg|opus|mp3|wav)", pm.group(2), re.I):
                    path_like = True
                    break
            if not path_like:
                continue
            for pm in pairs:
                k, v = pm.group(1), pm.group(2)
                if not re.search(r"\.(ogg|opus|mp3|wav)", v, re.I):
                    continue
                out[dname + '["' + k + '"]'] = v
    return out


def lastQuoted(line):
    dq = re.findall(r'"((?:[^"\\]|\\.)*)"', line)
    sq = re.findall(r"'((?:[^'\\]|\\.)*)'", line)
    if dq and sq:
        return (dq + sq)[-1]
    if dq:
        return dq[-1]
    if sq:
        return sq[-1]
    return None


def labelFromTid(tid):
    m = re.match(r"^(.+?)_[0-9a-f]{6,}", tid)
    if m:
        return m.group(1)
    return None


def scanTlMap(game_subdir):
    tl_map = {}
    tl_root = game_subdir / "tl"
    russian_dir = None
    if tl_root.exists():
        by_lower = {p.name.lower(): p for p in tl_root.iterdir() if p.is_dir()}
        for lang in RU_TL_FOLDERS:
            if lang in by_lower:
                russian_dir = by_lower[lang]
                break
    if russian_dir is None:
        print(game_subdir, "- \033[93m  папка перевода не найдена, парсинг только оригинала\033[0m")
        return tl_map, []
    label_counters = {}
    for f in sorted(russian_dir.rglob("*.rpy")):
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            m = RE_TL_LINE.match(lines[i])
            if not m:
                i += 1
                continue
            tid = m.group(1)
            label_name = labelFromTid(tid)
            i += 1

            translated_line = None
            while i < len(lines):
                stripped = lines[i].strip()
                if not stripped:
                    i += 1
                    continue
                if not stripped.startswith("#") and not stripped.startswith("translate"):
                    first_word = stripped.split()[0]
                    if first_word in STMT_KEYWORDS:
                        i += 1
                        continue
                    if '"""' in stripped or "'''" in stripped:
                        delim = '"""' if ('"""' in stripped) else "'''"
                        buf = stripped
                        if buf.count(delim) < 2:
                            j = i + 1
                            while j < len(lines):
                                buf += "\n" + lines[j]
                                if delim in lines[j]:
                                    break
                                j += 1
                            i = j + 1
                        else:
                            i += 1
                        translated_line = buf
                        break
                    if '"' in stripped or "'" in stripped:
                        translated_line = stripped
                        i += 1
                        break

                    i += 1
                    break
                if stripped.startswith("translate "):
                    break
                i += 1

            if label_name and translated_line:
                translated_text = lastQuoted(translated_line)
                if translated_text != None:
                    idx = label_counters.get(label_name, 0)
                    label_counters[label_name] = idx + 1
                    tl_map[(label_name, idx)] = translated_text

    return tl_map, sorted(russian_dir.rglob("*.rpy"))


def scanGame(game_dir):
    rpy_files, game_subdir = findRpyFiles(game_dir)
    if not rpy_files:
        return None
    labels = scanLabels(rpy_files)
    audio_map, all_vars = scanDefines(rpy_files)
    music_subscript_paths = scanMusicKeys(rpy_files)
    translations, tl_rpy_files = scanTlMap(game_subdir)

    game_name = game_dir.name
    game_subdir_path = game_subdir

    return {
        "rpy_files": rpy_files,
        "labels": labels,
        "audio_map": audio_map,
        "all_vars": all_vars,
        "music_subscript_paths": music_subscript_paths,
        "translations": translations,
        "tl_rpy_files": tl_rpy_files,
        "game_name": game_name,
        "game_subdir": game_subdir_path,
    }
