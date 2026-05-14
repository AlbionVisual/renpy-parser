import csv
import re
from pathlib import Path
from consts import (
    AUDIOS_DIR,
    BASE_INDENT,
    FIELDNAMES,
    STMT_KEYWORDS,
    RAW_GAMES,
    FIELDNAMESINORDER,
)

RE_PLAY = re.compile(
    r'^\s*(?:play|queue)\s+(music|sound|audio|sfx|music2|bgs)\s+'
    r'(?:\[\s*)?'
    r'(?:"([^"]+)"|(\w+))'
    r'(.*)',
    re.IGNORECASE,
)
RE_SUBSCRIPT_TAIL = re.compile(
    r'^\s*\[\s*["\']([^"\']+)["\']\s*\]',
)
_AUDIO_FILE_EXT = (".ogg", ".opus", ".mp3", ".wav")
RE_DOLLAR_PLAY_MUSIC = re.compile(
    r'^\$\s*play_music\s*\(\s*["\']([^"\']+)["\'](.*)$',
    re.IGNORECASE,
)
RE_STOP = re.compile(
    r"^\s*stop\s+(music|sound|audio|sfx|music2|bgs)", re.IGNORECASE)

RE_ZERO_INDENT_PY_HEADER = re.compile(
    r"^(if|elif|while|for)\s",
    re.I,
)
RE_ZERO_INDENT_ELSE = re.compile(r"^else\s*:", re.I)

RE_JUMP = re.compile(r"^(\s*)jump\s+([\w.]+)")
RE_CALL = re.compile(r"^(\s*)call\s+([\w.]+)")
RE_RETURN = re.compile(r"^(\s*)return\b")
RE_LABEL = re.compile(r"^label\s+([\w.]+)\s*(?:\(.*\))?\s*:")
RE_LABEL_LINE = re.compile(r"^\s*label\s+([\w.]+)\s*(?:\(.*\))?\s*:")
RE_SCREEN = re.compile(r"^\s*screen\s+\w+")
RE_PAUSE = re.compile(r"^\s*(?:pause|\$\s*renpy\.pause)")
RE_PY_JUMP = re.compile(r"renpy\.jump\(\s*['\"]([\w.]+)['\"]\s*\)")
RE_PY_CALL = re.compile(r"renpy\.call\(\s*['\"]([\w.]+)['\"]\s*\)")

RE_NOLOOP = re.compile(r"\bnoloop\b", re.IGNORECASE)

RE_DECISION_LINE = re.compile(r"^\".*\":$")
RE_TL_BLOCK_HEADER = re.compile(
    r"^\s*translate\s+\w+\s+([\w.]+)\s*:\s*$", re.IGNORECASE)
RE_TL_ID_TO_LABEL = re.compile(r"^(.+?)_[0-9a-f]{6,}$")

RE_CHAR_CALL_DIALOGUE = re.compile(
    r"^([\w.]+)\s*\([^)]*\)\s+(?:\"((?:[^\"\\]|\\.)*)\"|'((?:[^'\\]|\\.)*)')\s*$",
)

RE_JUMP_IN_EXPR = re.compile(
    r"""Jump\s*\(\s*['\"]([\w.]+)['\"]\s*\)""",
    re.IGNORECASE,
)

NO_MUSIC = "NO_MUSIC"

def playAudioRaw(m_play):
    path_str = m_play.group(2)
    var_name = m_play.group(3)
    tail = m_play.group(4) or ""
    if path_str:
        return path_str
    if var_name:
        st = RE_SUBSCRIPT_TAIL.match(tail)
        if st:
            return var_name + '["' + st.group(1) + '"]'
        return var_name
    return ""

RENPY_SCREEN_LINE_PREFIXES = frozenset(
    {
        "ground",
        "idle",
        "hover",
        "selected_idle",
        "selected_hover",
        "insensitive",
        "activate_sound",
        "hovered",
        "unhovered",
        "imagemap",
        "hotspot",
        "hotspotbutton",
        "focus_mask",
        "child_size",
    }
)

def isLooping(tail):
    if not tail:
        return True
    if re.search(r"\bloop\s*=\s*False\b", tail, re.IGNORECASE):
        return False
    return not RE_NOLOOP.search(tail)

def lineIndent(line):
    return len(line) - len(line.lstrip())

def parseDialogue(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("$"):
        return None, None

    if RE_DECISION_LINE.match(stripped):
        return None, None

    first_word = stripped.split()[0].rstrip('"\'' )
    if first_word in STMT_KEYWORDS:
        return None, None

    if first_word in RENPY_SCREEN_LINE_PREFIXES:
        return None, None

    if "(" in first_word and first_word not in ("_",):
        m_cc = RE_CHAR_CALL_DIALOGUE.match(stripped)
        if m_cc:
            tx = m_cc.group(2) if m_cc.group(2) is not None else m_cc.group(3)
            return m_cc.group(1), tx if tx is not None else ""
        return None, None

    if "'''" in stripped or '"""' in stripped:
        delim = "'''" if "'''" in stripped else '"""'
        text_start = stripped.find(delim)
        text_end = stripped.rfind(delim)
        if text_start >= 0 and text_end > text_start:
            text = stripped[text_start + len(delim):text_end]
            speaker = None
            before = stripped[:text_start].strip()
            if before:
                tokens = before.split()
                speaker = tokens[0]
            return speaker, text

    quote_char = '"'
    if '"' not in stripped:
        if "'" not in stripped:
            return None, None
        quote_char = "'"

    depth = 0
    text_start = -1
    text_end = -1
    i = len(stripped) - 1
    while i >= 0:
        if stripped[i] == quote_char:
            if depth == 0:
                text_end = i
                depth = 1
            else:
                if i == 0 or stripped[i - 1] != '\\':
                    text_start = i
                    break
        i -= 1

    if text_start < 0 or text_end < 0 or text_start >= text_end:
        return None, None

    text = stripped[text_start + 1:text_end]

    tl = text.strip().lower()
    if "/" in tl and re.search(
        r"\.(png|jpe?g|gif|webp|bmp|ogg|opus|mp3|wav)(\b|$)", tl, re.I
    ):
        return None, None

    speaker = None
    before = stripped[:text_start].strip()
    if before:
        maybe_speaker_part = before.rstrip('"\'' ).strip()
        last_q2 = maybe_speaker_part.rfind(quote_char)
        first_q = maybe_speaker_part.find(quote_char)
        if first_q >= 0 and last_q2 > first_q:
            speaker = maybe_speaker_part[first_q + 1:last_q2]
        else:
            tokens = before.split()
            speaker = tokens[0]

    return speaker, text

def readFileLines(rpy_files):
    file_lines = {}
    for f in rpy_files:
        text = Path(f).read_text(encoding="utf-8-sig", errors="replace")
        raw_lines = text.split("\n")

        out = list(raw_lines)
        multiline_delim = None
        start_idx = None
        buf = ""

        def openMerge(i, delim):
            nonlocal multiline_delim, start_idx, buf
            multiline_delim = delim
            start_idx = i
            buf = out[i]

        def closeMerge(i):
            nonlocal multiline_delim, start_idx, buf
            if start_idx is not None:
                out[start_idx] = buf
            for k in range(start_idx + 1, i + 1):
                out[k] = ""
            multiline_delim = None
            start_idx = None
            buf = ""

        for i, line in enumerate(raw_lines):
            if multiline_delim is None:
                if '"""' in line or "'''" in line:
                    dq = line.find('"""')
                    sq = line.find("'''")
                    delim = '"""' if dq != -1 and (sq == -1 or dq < sq) else "'''"
                    openMerge(i, delim)
                    if buf.count(delim) >= 2:
                        multiline_delim = None
                        start_idx = None
                        buf = ""
                    continue

                if line.count('"') == 1:
                    openMerge(i, '"')
                    continue

                continue

            buf += "\n" + line
            if multiline_delim in ('"""', "'''"):
                if buf.count(multiline_delim) >= 2:
                    closeMerge(i)
                continue

            if multiline_delim == '"':
                if buf.count('"') >= 2:
                    closeMerge(i)
                continue

        if multiline_delim is not None and start_idx is not None:
            out[start_idx] = buf
            for k in range(start_idx + 1, len(out)):
                out[k] = ""

        file_lines[str(f)] = out

    return file_lines

def fillLabelBodies(scan_data):
    labels = scan_data["labels"]
    rpy_files = scan_data["rpy_files"]
    file_lines = readFileLines(rpy_files)

    file_label_starts = {}
    for label_name, info in labels.items():
        fpath = str(info["file"])
        line_no = info["line"]
        if fpath not in file_label_starts:
            file_label_starts[fpath] = []
        file_label_starts[fpath].append((line_no, label_name))

    for fpath in file_label_starts:
        file_label_starts[fpath].sort()

    for label_name, info in labels.items():
        fpath = str(info["file"])
        start_line = info["line"]
        lines = file_lines[fpath]
        starts = file_label_starts[fpath]

        idx = next(i for i, (ln, _) in enumerate(starts) if ln == start_line)
        if idx + 1 < len(starts):
            end_line = starts[idx + 1][0]
            next_label = starts[idx + 1][1]
        else:
            end_line = len(lines)
            next_label = None

        body = []
        in_screen_block = False
        screen_block_indent = 0
        for i in range(start_line, end_line):
            if i >= len(lines):
                break
            raw_line = lines[i]
            if RE_SCREEN.match(raw_line):
                in_screen_block = True
                screen_block_indent = lineIndent(raw_line)
                continue
            if in_screen_block:
                if not raw_line.strip():
                    continue
                if lineIndent(raw_line) > screen_block_indent:
                    continue
                in_screen_block = False
            body.append((i + 1, raw_line))

        info["body"] = body
        base_indent = BASE_INDENT
        min_indent = None
        for _, raw_line in body[1:]:
            s = raw_line.strip()
            if not s or s.startswith("#"):
                continue
            ind = lineIndent(raw_line)
            if ind > 0 and (min_indent is None or ind < min_indent):
                min_indent = ind
        if min_indent is not None:
            base_indent = min_indent
        info["base_indent"] = base_indent
        info["next_label"] = next_label
        info["file_str"] = Path(fpath).name
        info["file_full"] = fpath

    return scan_data

def gameRelPath(abs_path, game_name, game_subdir):
    p = Path(abs_path)
    try:
        rel = p.relative_to(RAW_GAMES)
        return "raw-games/" + str(rel)
    except ValueError:
        return str(abs_path)

def normRel(val):
    if not val:
        return ""
    s = val.strip().strip('"').strip("'")
    return s.replace("\\", "/")

def stemExt(path):
    for e in _AUDIO_FILE_EXT:
        if path.lower().endswith(e):
            return path[: -len(e)], e
    return None, None

def fileInPaths(game_subdir, game_name, rel):
    if not rel or ".." in rel.split("/"):
        return False
    if game_subdir:
        if (Path(game_subdir) / rel).is_file():
            return True
    if game_name and AUDIOS_DIR:
        if (AUDIOS_DIR / game_name / rel).is_file():
            return True
    return False

def pickResolvedRel(game_subdir, game_name, rel):
    r = normRel(rel)
    if not r:
        return None
    candidates = []
    candidates.append(r)
    if not r.startswith("audio/"):
        candidates.append("audio/" + r)
    expanded = []
    for c in candidates:
        expanded.append(c)
        stem, cur_ext = stemExt(c)
        if stem is not None:
            for e in _AUDIO_FILE_EXT:
                if cur_ext and e.lower() == cur_ext.lower():
                    continue
                expanded.append(stem + e)
        else:
            for e in _AUDIO_FILE_EXT:
                expanded.append(c + e)
    seen = set()
    for c in expanded:
        if c in seen:
            continue
        seen.add(c)
        if fileInPaths(game_subdir, game_name, c):
            return c
    return None

def weirdLiteral(raw):
    s = raw.strip()
    return s.startswith("<")

def guessBareAudio(game_subdir, game_name, stem):
    st = stem.strip()
    if not st or "/" in st or "\\" in st or weirdLiteral(st):
        return None
    for ext in _AUDIO_FILE_EXT:
        rel = "audio/" + st + ext
        found = pickResolvedRel(game_subdir, game_name, rel)
        if found:
            return found
    return None

def clipAudioVal(audio_val, game_name, game_subdir):
    if not audio_val or audio_val == NO_MUSIC:
        return audio_val
    if audio_val.startswith("UNKNOWN:"):
        return audio_val
    if audio_val.startswith("raw-games/"):
        parts = audio_val.split("/")
        if "game" in parts:
            idx = parts.index("game")
            return "/".join(parts[idx+1:])
        elif len(parts) > 3:
            return "/".join(parts[3:])
        return audio_val
        
    return audio_val

def rowBgMedia(state, game_name, game_subdir):
    music = state.get("music", NO_MUSIC)
    if music != NO_MUSIC:
        return (
            clipAudioVal(music, game_name, game_subdir),
            state.get("music_loop", True),
            state.get("raw_music_line", ""),
        )
    bgs = state.get("bgs", NO_MUSIC)
    return (
        clipAudioVal(bgs, game_name, game_subdir),
        state.get("bgs_loop", True),
        state.get("raw_bgs_line", ""),
    )

def resolveAudio(
    raw_value,
    audio_map,
    all_vars,
    game_subdir=None,
    game_name=None,
    music_subscript_paths=None,
):
    if raw_value is None:
        return None, []

    warnings = []
    music_subscript_paths = music_subscript_paths or {}

    if weirdLiteral(raw_value):
        return NO_MUSIC, warnings

    if "/" in raw_value or "\\" in raw_value:
        rel = normRel(raw_value)
        found = pickResolvedRel(game_subdir, game_name, rel)
        if found:
            return found, warnings
        warnings.append("MISSING_FILE: " + rel)
        return "UNKNOWN:" + raw_value, warnings

    if re.search(r"\.(ogg|opus|mp3|wav)(\s|$)", raw_value, re.IGNORECASE):
        rel = "audio/" + raw_value.strip()
        found = pickResolvedRel(game_subdir, game_name, rel)
        if found:
            return found, warnings
        warnings.append("MISSING_FILE: " + rel)
        return "UNKNOWN:" + raw_value, warnings

    if raw_value in audio_map:
        rel = normRel(audio_map[raw_value])
        found = pickResolvedRel(game_subdir, game_name, rel)
        if found:
            return found, warnings
        warnings.append("MISSING_FILE: " + rel)
        return "UNKNOWN:" + raw_value, warnings

    key = "audio." + raw_value
    if key in all_vars:
        val = all_vars[key]["value"].strip("\"'")
        rel = normRel(val)
        found = pickResolvedRel(game_subdir, game_name, rel)
        if found:
            return found, warnings
        warnings.append("MISSING_FILE: " + rel)
        return "UNKNOWN:" + raw_value, warnings

    if raw_value in all_vars:
        val = all_vars[raw_value]["value"].strip("\"'")
        if "/" in val or "." in val:
            rel = normRel(val)
            found = pickResolvedRel(game_subdir, game_name, rel)
            if found:
                return found, warnings
            warnings.append("MISSING_FILE: " + rel)
            return "UNKNOWN:" + raw_value, warnings
        warnings.append(
            "UNRESOLVED_CHAIN: " + raw_value + " -> " + val
            + " (define at " + all_vars[raw_value]["file"]
            + ":" + str(all_vars[raw_value]["line"]) + ")"
        )
        return "UNKNOWN:" + raw_value, warnings

    if raw_value in music_subscript_paths:
        inner = music_subscript_paths[raw_value]
        resolved, w2 = resolveAudio(
            inner,
            audio_map,
            all_vars,
            game_subdir,
            game_name,
            None,
        )
        warnings.extend(w2)
        return resolved, warnings

    guessed = guessBareAudio(
        game_subdir, game_name, raw_value)
    if guessed:
        return guessed, warnings

    warnings.append("UNKNOWN_VAR: " + raw_value)
    return "UNKNOWN:" + raw_value, warnings

def preferLinear(n_labels, n_dialogue_rows):
    if n_labels < 25 or n_dialogue_rows <= 0:
        return False
    return (n_dialogue_rows / n_labels) < 2.2

def scanLinearGame(scan_data):
    rpy_files = sorted(scan_data["rpy_files"], key=lambda p: str(p).lower())
    file_lines = readFileLines(rpy_files)
    audio_map = scan_data["audio_map"]
    all_vars = scan_data["all_vars"]
    music_subscript_paths = scan_data.get("music_subscript_paths", {})
    translations = scan_data.get("translations", {})
    game_name = scan_data["game_name"]
    game_subdir = scan_data["game_subdir"]

    dataset = []
    warnings = []
    order = [0]
    state = {
        "music": NO_MUSIC,
        "music_loop": True,
        "raw_music_line": "",
        "bgs": NO_MUSIC,
        "bgs_loop": True,
        "raw_bgs_line": "",
        "pending_sfx": None,
        "sfx_invalidated": False,
    }
    label_dialogue_counters = {}
    cur_label = ""

    for f in rpy_files:
        fp = str(f)
        lines = file_lines[fp]
        file_rel = gameRelPath(fp, game_name, game_subdir)
        in_screen_block = False
        screen_block_indent = 0

        for line_no, raw_line in enumerate(lines, start=1):
            if RE_SCREEN.match(raw_line):
                in_screen_block = True
                screen_block_indent = lineIndent(raw_line)
                continue
            if in_screen_block:
                if not raw_line.strip():
                    continue
                if lineIndent(raw_line) > screen_block_indent:
                    continue
                in_screen_block = False

            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            m_lab = RE_LABEL_LINE.match(raw_line)
            if m_lab:
                cur_label = m_lab.group(1)
                continue

            m_play = RE_PLAY.match(raw_line)
            if m_play:
                channel = m_play.group(1).lower()
                path_str = m_play.group(2)
                var_name = m_play.group(3)
                tail = m_play.group(4) or ""

                raw = playAudioRaw(m_play)
                if not raw:
                    continue
                resolved, w = resolveAudio(
                    raw,
                    audio_map,
                    all_vars,
                    game_subdir,
                    game_name,
                    music_subscript_paths,
                )
                warnings.extend(w)

                if channel in ("music", "music2"):
                    state["music"] = resolved or NO_MUSIC
                    state["music_loop"] = isLooping(tail)
                    state["raw_music_line"] = stripped
                elif channel == "bgs":
                    state["bgs"] = resolved or NO_MUSIC
                    state["bgs_loop"] = isLooping(tail)
                    state["raw_bgs_line"] = stripped
                elif channel in ("sound", "sfx", "audio"):
                    full_sfx = clipAudioVal(
                        resolved, game_name, game_subdir)
                    state["pending_sfx"] = full_sfx
                    state["sfx_invalidated"] = False
                continue

            m_dollar_music = RE_DOLLAR_PLAY_MUSIC.match(stripped)
            if m_dollar_music:
                path_arg = m_dollar_music.group(1)
                tail_dm = m_dollar_music.group(2) or ""
                resolved, w = resolveAudio(
                    path_arg,
                    audio_map,
                    all_vars,
                    game_subdir,
                    game_name,
                    music_subscript_paths,
                )
                warnings.extend(w)
                state["music"] = resolved or NO_MUSIC
                state["music_loop"] = isLooping(tail_dm)
                state["raw_music_line"] = stripped
                continue

            m_stop = RE_STOP.match(raw_line)
            if m_stop:
                channel = m_stop.group(1).lower()
                if channel in ("music", "music2"):
                    state["music"] = NO_MUSIC
                    state["music_loop"] = True
                    state["raw_music_line"] = stripped
                elif channel == "bgs":
                    state["bgs"] = NO_MUSIC
                    state["bgs_loop"] = True
                    state["raw_bgs_line"] = ""
                elif channel in ("sound", "sfx", "audio"):
                    state["pending_sfx"] = None
                continue

            if RE_PAUSE.match(raw_line):
                state["sfx_invalidated"] = True
                continue

            speaker, text = parseDialogue(raw_line)
            if not text:
                continue

            order[0] += 1

            sfx_before = None
            if state["pending_sfx"] and not state["sfx_invalidated"]:
                sfx_before = state["pending_sfx"]
            state["pending_sfx"] = None
            state["sfx_invalidated"] = False

            music_full, bg_loop, raw_bg = rowBgMedia(
                state, game_name, game_subdir)

            lab = cur_label or "_"
            dl_idx = label_dialogue_counters.get(lab, 0)
            label_dialogue_counters[lab] = dl_idx + 1
            text_translated = translations.get((lab, dl_idx))

            dataset.append({
                FIELDNAMES["order in text"]: order[0],
                FIELDNAMES["who is speaking"]: speaker or "",
                FIELDNAMES["original text"]: text,
                FIELDNAMES["text from translations"]: text_translated,
                FIELDNAMES["path to bg music"]: music_full,
                FIELDNAMES["is music looped"]: bg_loop,
                FIELDNAMES["line with music start"]: raw_bg,
                FIELDNAMES["path to sound effects"]: sfx_before,
                FIELDNAMES["label of separable game parts"]: lab,
                FIELDNAMES["file containing text"]: file_rel,
                FIELDNAMES["line in file"]: line_no,
                FIELDNAMES["game name"]: game_name
            })

    return dataset, warnings

def jumpTargets(rpy_files):
    targets = set()
    for f in rpy_files:
        try:
            txt = Path(f).read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        for m in RE_JUMP_IN_EXPR.finditer(txt):
            targets.add(m.group(1))
    return targets

def traceGame(scan_data, start_label="start"):
    scan_data = fillLabelBodies(scan_data)
    audio_map = scan_data["audio_map"]
    all_vars = scan_data["all_vars"]
    music_subscript_paths = scan_data.get("music_subscript_paths", {})
    labels = scan_data["labels"]
    translations = scan_data.get("translations", {})
    game_name = scan_data["game_name"]
    game_subdir = scan_data["game_subdir"]

    if start_label not in labels:
        print("  label '" + start_label + "' не найден")
        return [], []

    dataset = []
    warnings = []
    visited = set()
    order = [0]
    label_dialogue_counters = {}

    state = {
        "music": NO_MUSIC,
        "music_loop": True,
        "raw_music_line": "",
        "bgs": NO_MUSIC,
        "bgs_loop": True,
        "raw_bgs_line": "",
        "pending_sfx": None,
        "sfx_invalidated": False,
    }

    def tidToLabel(tid):
        m = RE_TL_ID_TO_LABEL.match(tid)
        if m:
            return m.group(1)
        return tid

    def scanTlFiles():
        tl_files = scan_data.get("tl_rpy_files") or []
        if not tl_files:
            return

        for f in tl_files:
            file_full = str(f)
            file_rel = gameRelPath(file_full, game_name, game_subdir)
            try:
                lines = Path(f).read_text(
                    encoding="utf-8-sig", errors="replace").split("\n")
            except Exception:
                continue

            cur_label = None
            label_dl_idx = 0
            for i, line in enumerate(lines, start=1):
                m = RE_TL_BLOCK_HEADER.match(line)
                if m:
                    tid = m.group(1)
                    cur_label = tidToLabel(tid)
                    label_dl_idx = 0
                    continue

                if cur_label is None:
                    continue

                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                m_play = RE_PLAY.match(line)
                if m_play:
                    channel = m_play.group(1).lower()
                    path_str = m_play.group(2)
                    var_name = m_play.group(3)
                    tail = m_play.group(4) or ""

                    raw = playAudioRaw(m_play)
                    if not raw:
                        continue
                    resolved, w = resolveAudio(
                        raw,
                        audio_map,
                        all_vars,
                        game_subdir,
                        game_name,
                        music_subscript_paths,
                    )
                    warnings.extend(w)

                    if channel in ("music", "music2"):
                        state["music"] = resolved or NO_MUSIC
                        state["music_loop"] = isLooping(tail)
                        state["raw_music_line"] = stripped
                    elif channel == "bgs":
                        state["bgs"] = resolved or NO_MUSIC
                        state["bgs_loop"] = isLooping(tail)
                        state["raw_bgs_line"] = stripped
                    elif channel in ("sound", "sfx", "audio"):
                        full_sfx = clipAudioVal(
                            resolved, game_name, game_subdir)
                        state["pending_sfx"] = full_sfx
                        state["sfx_invalidated"] = False
                    continue

                m_dollar_music = RE_DOLLAR_PLAY_MUSIC.match(stripped)
                if m_dollar_music:
                    path_arg = m_dollar_music.group(1)
                    tail_dm = m_dollar_music.group(2) or ""
                    resolved, w = resolveAudio(
                        path_arg,
                        audio_map,
                        all_vars,
                        game_subdir,
                        game_name,
                        music_subscript_paths,
                    )
                    warnings.extend(w)
                    state["music"] = resolved or NO_MUSIC
                    state["music_loop"] = isLooping(tail_dm)
                    state["raw_music_line"] = stripped
                    continue

                m_stop = RE_STOP.match(line)
                if m_stop:
                    channel = m_stop.group(1).lower()
                    if channel in ("music", "music2"):
                        state["music"] = NO_MUSIC
                        state["music_loop"] = True
                        state["raw_music_line"] = stripped
                    elif channel == "bgs":
                        state["bgs"] = NO_MUSIC
                        state["bgs_loop"] = True
                        state["raw_bgs_line"] = ""
                    elif channel in ("sound", "sfx", "audio"):
                        state["pending_sfx"] = None
                    continue

                if RE_PAUSE.match(line):
                    state["sfx_invalidated"] = True
                    continue

                speaker, text = parseDialogue(line)
                if text:
                    order[0] += 1

                    sfx_before = None
                    if state["pending_sfx"] and not state["sfx_invalidated"]:
                        sfx_before = state["pending_sfx"]
                    state["pending_sfx"] = None
                    state["sfx_invalidated"] = False

                    music_full, bg_loop, raw_bg = rowBgMedia(
                        state, game_name, game_subdir)

                    text_translated = translations.get((cur_label, label_dl_idx))
                    label_dl_idx += 1

                    dataset.append({
                        FIELDNAMES["order in text"]: order[0],
                        FIELDNAMES["who is speaking"]: speaker or "",
                        FIELDNAMES["original text"]: text,
                        FIELDNAMES["text from translations"]: text_translated,
                        FIELDNAMES["path to bg music"]: music_full,
                        FIELDNAMES["is music looped"]: bg_loop,
                        FIELDNAMES["line with music start"]: raw_bg,
                        FIELDNAMES["path to sound effects"]: sfx_before,
                        FIELDNAMES["label of separable game parts"]: cur_label,
                        FIELDNAMES["file containing text"]: file_rel,
                        FIELDNAMES["line in file"]: i,
                        FIELDNAMES["game name"]: game_name
                    })

    def walkLabel(label_name):
        if label_name not in labels:
            warnings.append("MISSING_LABEL: " + label_name)
            return "missing"
        if label_name in visited:
            return "visited"
        visited.add(label_name)

        info = labels[label_name]
        body = info.get("body", [])
        base_indent = info.get("base_indent", BASE_INDENT)
        src_file = info.get("file_str", "?")
        file_full = info.get("file_full", "?")
        next_label = info.get("next_label")

        file_rel = gameRelPath(file_full, game_name, game_subdir)

        skip_next_label_fallthrough = False
        flat_py_active = False
        for line_no, line in body:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            zi = lineIndent(line)
            if zi == 0:
                if (
                    RE_ZERO_INDENT_PY_HEADER.match(stripped)
                    or RE_ZERO_INDENT_ELSE.match(stripped)
                ):
                    flat_py_active = True
                elif not RE_LABEL_LINE.match(line):
                    flat_py_active = False

            indent = zi
            is_nested = indent > base_indent or (
                indent == base_indent and flat_py_active
            )

            m_play = RE_PLAY.match(line)
            if m_play:
                channel = m_play.group(1).lower()
                path_str = m_play.group(2)
                var_name = m_play.group(3)
                tail = m_play.group(4) or ""

                raw = playAudioRaw(m_play)
                if not raw:
                    continue
                resolved, w = resolveAudio(
                    raw,
                    audio_map,
                    all_vars,
                    game_subdir,
                    game_name,
                    music_subscript_paths,
                )
                warnings.extend(w)

                if channel in ("music", "music2"):
                    state["music"] = resolved or NO_MUSIC
                    state["music_loop"] = isLooping(tail)
                    state["raw_music_line"] = stripped
                elif channel == "bgs":
                    state["bgs"] = resolved or NO_MUSIC
                    state["bgs_loop"] = isLooping(tail)
                    state["raw_bgs_line"] = stripped
                elif channel in ("sound", "sfx", "audio"):
                    full_sfx = clipAudioVal(
                        resolved, game_name, game_subdir)
                    state["pending_sfx"] = full_sfx
                    state["sfx_invalidated"] = False
                continue

            m_dollar_music = RE_DOLLAR_PLAY_MUSIC.match(stripped)
            if m_dollar_music:
                path_arg = m_dollar_music.group(1)
                tail_dm = m_dollar_music.group(2) or ""
                resolved, w = resolveAudio(
                    path_arg,
                    audio_map,
                    all_vars,
                    game_subdir,
                    game_name,
                    music_subscript_paths,
                )
                warnings.extend(w)
                state["music"] = resolved or NO_MUSIC
                state["music_loop"] = isLooping(tail_dm)
                state["raw_music_line"] = stripped
                continue

            m_stop = RE_STOP.match(line)
            if m_stop:
                channel = m_stop.group(1).lower()
                if channel in ("music", "music2"):
                    state["music"] = NO_MUSIC
                    state["music_loop"] = True
                    state["raw_music_line"] = stripped
                elif channel == "bgs":
                    state["bgs"] = NO_MUSIC
                    state["bgs_loop"] = True
                    state["raw_bgs_line"] = ""
                elif channel in ("sound", "sfx", "audio"):
                    state["pending_sfx"] = None
                continue

            if RE_PAUSE.match(line):
                state["sfx_invalidated"] = True
                continue

            m_jump = RE_JUMP.match(line)
            if m_jump:
                target = m_jump.group(2)
                if is_nested:
                    walkLabel(target)
                    continue
                else:
                    walkLabel(target)
                    return "jumped"

            m_call = RE_CALL.match(line)
            if m_call:
                target = m_call.group(2)
                if target == "screen":
                    if not is_nested:
                        skip_next_label_fallthrough = True
                        break
                    continue
                walkLabel(target)
                continue

            m_py_jump = RE_PY_JUMP.search(line)
            if m_py_jump:
                target = m_py_jump.group(1)
                if is_nested:
                    walkLabel(target)
                    continue
                walkLabel(target)
                return "jumped"

            m_py_call = RE_PY_CALL.search(line)
            if m_py_call:
                target = m_py_call.group(1)
                walkLabel(target)
                continue

            m_ret = RE_RETURN.match(line)
            if m_ret:
                if not is_nested:
                    return "returned"
                continue

            speaker, text = parseDialogue(line)
            if text:
                order[0] += 1

                sfx_before = None
                if state["pending_sfx"] and not state["sfx_invalidated"]:
                    sfx_before = state["pending_sfx"]
                state["pending_sfx"] = None
                state["sfx_invalidated"] = False

                music_full, bg_loop, raw_bg = rowBgMedia(
                    state, game_name, game_subdir)

                dl_idx = label_dialogue_counters.get(label_name, 0)
                label_dialogue_counters[label_name] = dl_idx + 1
                text_translated = translations.get((label_name, dl_idx))

                dataset.append({
                    FIELDNAMES["order in text"]: order[0],
                    FIELDNAMES["who is speaking"]: speaker or "",
                    FIELDNAMES["original text"]: text,
                    FIELDNAMES["text from translations"]: text_translated,
                    FIELDNAMES["path to bg music"]: music_full,
                    FIELDNAMES["is music looped"]: bg_loop,
                    FIELDNAMES["line with music start"]: raw_bg,
                    FIELDNAMES["path to sound effects"]: sfx_before,
                    FIELDNAMES["label of separable game parts"]: label_name,
                    FIELDNAMES["file containing text"]: file_rel,
                    FIELDNAMES["line in file"]: line_no,
                    FIELDNAMES["game name"]: game_name
                })

        if next_label and not skip_next_label_fallthrough:
            walkLabel(next_label)
            return "fallthrough"

        return "end"

    walkLabel(start_label)
    jump_targets = jumpTargets(scan_data["rpy_files"])
    for jt in sorted(jump_targets):
        if jt in labels and jt not in visited:
            walkLabel(jt)

    n_lab = len(labels)
    n_d = len(dataset)
    if n_d > 0 and preferLinear(n_lab, n_d):
        ds_lin, w_lin = scanLinearGame(scan_data)
        if len(ds_lin) >= n_d:
            dataset = ds_lin
            warnings = warnings + w_lin
            print(
                "  линейный проход (много меток / ветвлений): строк",
                len(dataset),
                "вместо графовой трассировки",
                n_d,
            )

    if not dataset:
        scanTlFiles()
    return dataset, warnings

def saveCsv(dataset, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMESINORDER)
        writer.writeheader()
        writer.writerows(dataset)
    print("  CSV записан:", output_path, "строк:", len(dataset))
