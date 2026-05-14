## Цикл обработки музыки:

#### Синхронизация игр с бд
```
python music_scripts/parse_folder_to_db.py sync_games_db
```
#### Синхронизация таблиц в бд
```
python music_scripts/parse_folder_to_db.py sync_columns
```
#### Вместо двух выше
```
python music_scripts/fix_music_paths_from_audios.py
```
#### Ручной разбор UNKNOWN / несопоставленных путей (консоль)
```
python3 music_scripts/resolve_unknown_music_interactive.py
```
Опции: `--only-unknown-prefix`, `--game <имя>`, `--dry-run`, `--max-hints 12`.

#### Диагностика после синка
```
python music_scripts/inspect_confused_music.py
```
#### Удаление записей, которые не исопльзуются в тектсе (удаляет и анализ тоже)
```
python music_scripts/prune_unused_music_data.py
```
#### Анализ музыки
```
python3 music_scripts/AudioEnricher.py
```