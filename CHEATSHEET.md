# Cheatsheet: Postgres, CSV, размер чанков

## PostgreSQL через Python

### Подключение (DSN)

Обычно используют DSN строку вида:

`postgresql://USER:PASSWORD@HOST:PORT/DBNAME`

или набор переменных окружения (`PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`).

### Минимальный пример с `psycopg` (psycopg3)

```python
import psycopg

conn = psycopg.connect("postgresql://user:pass@localhost:5432/dbname")
cur = conn.cursor()
cur.execute("select 1")
print(cur.fetchone())
conn.close()
```

### Пакетная вставка (много строк)

Если вставляешь много строк, обычно делают batch insert (или `COPY`, если хочется максимальной скорости).

```python
import psycopg

rows = [
    ("game1", "file1.rpy", 10, "hello"),
    ("game1", "file1.rpy", 11, "world"),
]

conn = psycopg.connect("postgresql://user:pass@localhost:5432/dbname")
cur = conn.cursor()
cur.executemany(
    "insert into dialogue_lines (game, file, line, text) values (%s, %s, %s, %s) on conflict do nothing",
    rows,
)
conn.commit()
conn.close()
```

### Быстрая загрузка через `COPY` (когда уже есть CSV)

Вариант через Python: готовишь CSV, затем `copy_expert`/`copy` (в зависимости от версии драйвера).
Если не хочется помнить детали API драйвера, часто проще использовать `psql`/`\\copy` из bash (см. ниже).

## PostgreSQL через bash (`psql`)

### Подключение

Через DSN:

```bash
psql "postgresql://user:pass@localhost:5432/dbname"
```

Через env (удобно для скриптов):

```bash
export PGHOST=localhost
export PGPORT=5432
export PGUSER=user
export PGPASSWORD=pass
export PGDATABASE=dbname
psql
```

### Выполнить SQL из bash без интерактива

```bash
psql "postgresql://user:pass@localhost:5432/dbname" -c "select 1;"
```

### Импорт CSV в таблицу

Если CSV лежит на машине, где запускается `psql`, то проще всего `\\copy`:

```bash
psql "postgresql://user:pass@localhost:5432/dbname" -c "\\copy dialogue_lines from 'data.csv' with (format csv, header true)"
```

Если нужны конкретные колонки и порядок:

```bash
psql "postgresql://user:pass@localhost:5432/dbname" -c "\\copy dialogue_lines(game,file,line,text) from 'data.csv' with (format csv, header true)"
```
### Экспорт таблицы в CSV

```bash
psql "postgresql://user:pass@localhost:5432/dbname" -c "\\copy (select * from dialogue_lines order by game, file, line) to 'out.csv' with (format csv, header true)"
```

## PostgreSQL как сервис (запуск / стоп / статус / проверка)

Команды зависят от того, как установлен Postgres (systemd service или Docker). Ниже — самый частый случай для WSL/Linux с systemd.

### systemd (`systemctl`)

```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
sudo systemctl stop postgresql
sudo systemctl restart postgresql
```

Иногда сервис называется с версией, например `postgresql@16-main`:

```bash
sudo systemctl status postgresql@16-main
```

### Проверить, что Postgres “жив”

Проверка готовности:

```bash
pg_isready
```

Проверка, что порт слушается:

```bash
ss -ltnp | grep 5432
```

Проверка подключением:

```bash
psql "postgresql://user:pass@localhost:5432/dbname" -c "select now();"
```

## CSV в Python

### Читать CSV как dict

```python
import csv

with open("in.csv", "r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        print(row["text"])
        break
```

### Писать CSV из dict-строк

```python
import csv

rows = [
    {"game": "g", "file": "f", "line": "1", "text": "hello"},
    {"game": "g", "file": "f", "line": "2", "text": "world"},
]

with open("out.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["game", "file", "line", "text"])
    w.writeheader()
    w.writerows(rows)
```

## `.env` в Python-скриптах

### Вариант без зависимостей (самостоятельно загрузить в `os.environ`)

```python
import os

def load_dotenv(path=".env"):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

load_dotenv()
print(os.environ.get("PGHOST"))
```

Ограничения такого мини-загрузчика: не умеет кавычки, `export KEY=...`, экранирование и многострочные значения. Зато не требует пакетов.

### Вариант с пакетом `python-dotenv` (если захочешь удобства)

```python
from dotenv import load_dotenv

load_dotenv()
```

## Как трекать размер записи в CSV (чанк до лимита)

Ниже три рабочих подхода. Общая идея: считать **реальные байты в UTF-8**, а не “количество символов”.

### Подход 1: считать байты каждой строки CSV до записи

Генерируешь одну CSV-строку, меряешь `len(s.encode("utf-8"))`, и решаешь добавлять или нет.

```python
import csv
import io

def csv_line_bytes(fieldnames, row):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    w.writerow(row)
    s = buf.getvalue()
    return len(s.encode("utf-8")), s

fieldnames = ["game", "file", "line", "text"]
limit = 1000000
total = 0

row = {"game": "g", "file": "f", "line": 1, "text": "hello"}
b, s = csv_line_bytes(fieldnames, row)
if total + b <= limit:
    total += b
print("bytes", total)
```

### Подход 2: считать размер файла по `tell()` (когда пишешь в файл)

Это удобно, если ты пишешь сразу в реальный файл.

```python
import csv

fieldnames = ["game", "file", "line", "text"]
limit = 1000000

with open("out.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    before = f.tell()
    w.writerow({"game": "g", "file": "f", "line": 1, "text": "hello"})
    after = f.tell()
    if after > limit:
        print("limit_exceeded", after)
    else:
        print("ok", after, "delta", after - before)
```

Нюанс: если лимит превышен, “откатить” последнюю строку сложнее (обычно решают тем, что считают размер *до* записи, как в Подходе 1).

### Подход 3: считать размер уже готового фрагмента CSV

Если у тебя уже есть список строк `list[str]` и ты хочешь оценить размер:

```python
lines = ["a,b,c\n", "1,2,3\n"]
data = "".join(lines).encode("utf-8")
print("bytes", len(data))
```

