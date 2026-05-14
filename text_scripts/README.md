## Цикл обработки текста (БД `text_musics_rel` → ML)

Нужен `.env` с `pghost`. Исходные колонки парсера — `text_orig` / `text_translated`; рабочая колонка для моделей — **`text1`**.

#### Заполнить `text1` (интерактивно по играм)

```
python3 text_scripts/fill_text1_interactive.py
```

#### Заполнить `text1` без диалога

```
python3 text_scripts/fill_text1_auto.py --prefer smart
```

#### Нормализация скобочных форм `[вариант1_вариант2]` в `text1` (интерактивно)

```
python3 text_scripts/choose_bracket_forms.py
```

#### Очистка `text1` (теги, мусор, плейсхолдеры и т.п.)

```
python3 text_scripts/clean_text1.py
```

#### Копия строк в таблицу под RuBERT (`text_musics_rubert`)

```
python3 text_scripts/ml_scripts/sync_text_musics_rubert_table.py
```

Полный сброс содержимого целевой таблицы перед вставкой: добавьте **`--truncate`** (см. `--help`).

#### Эмбеддинги RuBERT в БД (`text_analized` и др.)

```
python3 text_scripts/ml_scripts/run_rubert_encode.py --from-db --write-db --device auto
```

Перекодировать все строки, даже с уже заполненным эмбеддингом: **`--reencode-all`**. См. также `--help` (`--batch-size`, `--db-column`, `--model`, …).

#### Экспорт датасета в файлы (Colab / без Postgres на чтении)

```
python3 text_scripts/ml_scripts/export_dataset_v1.py --out-dir ./dataset_v1
```

#### Обучение / манифест (из `text_scripts/ml_scripts/`)

Скрипты вида `train_rubert_to_music_*.py`, оркестратор:

```
python3 text_scripts/ml_scripts/run_manifest.py --manifest ./manifest.jsonl --base-run-dir ./runs --dataset-dir ./dataset_v1 --resume
```

#### SQL-шпаргалка

```
text_scripts/sql_queries.md
```
