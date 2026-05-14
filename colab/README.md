# Colab runner

## Что должно лежать на Google Drive

После `drive.mount("/content/drive")` личный диск виден как **`/content/drive/MyDrive/`** — это «корень» веб-интерфейса «Мой диск», не отдельная папка с именем MyDrive.

Создай на **Мой диск** такую структуру (имена можно другие, но тогда поправь переменные в `runner.ipynb`):

| Путь в Drive (в браузере) | Путь в Colab | Обязательно |
|---------------------------|--------------|-------------|
| `Мой диск/dataset_v1/`   | `/content/drive/MyDrive/dataset_v1` | да |
| `Мой диск/rubert_runs/`  | `/content/drive/MyDrive/rubert_runs` | да (можно пустая папка) |
| `Мой диск/manifest.jsonl` | `/content/drive/MyDrive/manifest.jsonl` | да |

### Папка `dataset_v1/` (результат локального `export_dataset_v1.py`)

Должны быть как минимум:

- `meta.csv`
- `x.npy`
- `y.npy`
- `y_columns.json`
- `candidates/index.json` и файлы `candidates/*.npz` (для retrieval и для autoreg)

### Файл `manifest.jsonl`

Это **текстовый файл**, который ты **сам создаёшь** и кладёшь на Drive (репозиторий его не содержит). Один запуск = одна строка JSON.

**Интерактивная сборка (после `git pull` на своей машине):**

```bash
cd /path/to/renpy-parser
python3 text_scripts/ml_scripts/build_manifest_interactive.py -o ./manifest.jsonl
```

Скрипт спросит общие параметры (device, epochs, split, сиды, val), затем в цикле — какую модель добавить в очередь и `run_id`; порядок добавления = порядок строк в файле. Сгенерированный `manifest.jsonl` загрузи на Drive (или укажи путь сразу под Google Drive для Linux).

Флаги: `-o /путь/manifest.jsonl`, `--append` — дописать в конец существующего файла.

### Сборщик manifest прямо в Colab

**Да, можно.** После `drive.mount`, `git clone` репозитория в `/content/repo` и при необходимости `pip install …` запускай тот же скрипт; главное — записать файл на **Drive**, чтобы он не пропал после отключения сессии:

```bash
cd /content/repo
python3 text_scripts/ml_scripts/build_manifest_interactive.py \
  -o "/content/drive/MyDrive/manifest.jsonl"
```

**Интерактивные вопросы (`input`)** в Colab надёжнее всего отвечать в **встроенном терминале**, а не в ячейке с `!`: слева **⌘/Ctrl+`** или меню **View → Terminal**, там же выполни команду выше — подсказки и ввод работают как в обычном терминале. В ячейке вида `!python …` интерактивный stdin часто **не подключается** к вводу.

Альтернатива без терминала: собрать `manifest.jsonl` **у себя на ПК** скриптом или вручную и **загрузить на Drive** — для Colab это то же самое.

---

## Git и Colab (подробнее)

| Идея | Пояснение |
|------|-----------|
| **Виртуалка временная** | Содержимое `/content` обычно **теряется**, когда сессия Colab отключается (таймаут, закрыл вкладку, «сброс» runtime). Не храни там единственную копию датасета или манифеста без Drive. |
| **Drive остаётся** | Всё под `/content/drive/MyDrive/…` — твой Google Диск, переживает перезапуски. Имеет смысл класть туда `dataset_v1/`, `manifest.jsonl`, `rubert_runs/`. |
| **Код на каждый сеанс** | Типично: **заново** `git clone` в `/content/repo` и `git pull` не нужен, пока не захочешь обновиться. Либо один раз клонировать **на Drive** (например `MyDrive/repos/renpy-parser`) и в ноутбуке делать `cd` туда + `git pull` — тогда не тянешь весь репо каждый раз с нулевой историей, но путь к репо в ячейках надо сменить с `/content/repo` на путь на Drive. |
| **Цикл разработки** | Меняешь код **локально** → `git commit` → `git push` → в Colab в каталоге репо **`git pull`** (или снова `git clone --depth 1`). Пока не запушил — Colab новую версию не увидит. |
| **Приватный GitHub** | **HTTPS:** в Colab можно `git clone https://<TOKEN>@github.com/user/repo.git` (токен с scope `repo`); токен не вставляй в ноутбук, который публикуешь. Удобнее **GitHub fine-grained PAT** или **SSH** (сложнее настроить в Colab). |
| **Ветки** | `git clone -b mybranch …` или после clone: `git checkout otherbranch && git pull`. |
| **Что не коммитить** | Секреты (`.env`, пароли БД), огромные `dataset_v1/` — у тебя в `.gitignore` уже есть `dataset_v1/`. |

Итого: **да, обновления кода = push с твоей машины + pull (или повторный clone) в Colab.** Манифест и данные живут на **Drive**; репозиторий в `/content/repo` — рабочая копия «на сессию», если не клонировал на Drive.

---

## Запуск

```json
{"run_id":"smoke_mlp","script":"text_scripts/ml_scripts/train_rubert_to_music_mlp.py","args":["--device","cuda","--epochs","2","--split","leave_games_out","--test-games","3","--held-out-games-seed","42","--val-split","tail","--val-fraction","0.1","--hidden","256,128","--batch-size","128","--no-progress"],"seed":42,"held_out_games_seed":42,"status":"pending"}
```

Как создать: Google Диск → **Создать** → **Google Таблицы** не нужны; проще **загрузить файл** с компьютера или в Colab в ячейке записать:

```python
%%writefile /content/drive/MyDrive/manifest.jsonl
{"run_id":"smoke_mlp","script":"text_scripts/ml_scripts/train_rubert_to_music_mlp.py","args":["--device","cuda","--epochs","1","--split","leave_games_out","--test-games","3","--held-out-games-seed","42","--seed","42","--val-split","tail","--val-fraction","0.1","--hidden","256,128","--batch-size","128","--no-progress"],"seed":42,"held_out_games_seed":42,"status":"pending"}

```

Проверка путей в Colab:

```python
from pathlib import Path
for p in [MANIFEST_PATH, DATASET_DIR, RUNS_DIR]:
    print(p, "exists" if Path(p).exists() else "MISSING")
```

Ошибка `FileNotFoundError: ... manifest.jsonl` значит: файл **не создан** на Drive, лежит **в другом каталоге** (не в корне «Мой диск»), или в `MANIFEST_PATH` опечатка.

---

## Запуск

1. Mount Google Drive.
2. Clone this repo (or upload a snapshot).
3. `pip install -r text_scripts/ml_scripts/requirements_colab.txt`
4. Убедись, что на Drive есть `dataset_v1/`, `manifest.jsonl`, папка под прогоны (`rubert_runs`).
5. Один job за один вызов (повторяй ячейку или вызывай в цикле):

`python -u text_scripts/ml_scripts/run_manifest.py --manifest /path/manifest.jsonl --base-run-dir /path/runs --dataset-dir /path/dataset_v1 --resume`

`--resume` подхватывает `failed` и сбрасывает зависший `running` без артефактов.

Артефакты каждого run: `metrics.json` или `per_column_metrics.csv`, плюс `stdout.log` и `config.json` в каталоге run на Drive.
