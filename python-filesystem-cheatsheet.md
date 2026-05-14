## Шпаргалка: файловая система в Python (чтение/запись/скан директорий)

### Чтение файлов

- **Прочитать весь файл строкой**

```python
text = open("a.txt", "r", encoding="utf-8").read()
print(text)
```

- **Прочитать весь файл построчно (список строк)**

```python
lines = open("a.txt", "r", encoding="utf-8").read().splitlines()
print(lines)
```

- **Читать построчно в цикле (экономит память на больших файлах)**

```python
f = open("a.txt", "r", encoding="utf-8")
for line in f:
    print(line.rstrip("\n"))
f.close()
```

- **Правильный вариант с автозакрытием (рекомендуется)**

```python
with open("a.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.rstrip("\n"))
```

### Запись файлов

- **Перезаписать файл целиком**

```python
with open("a.txt", "w", encoding="utf-8") as f:
    f.write("hello\n")
```

- **Дописать в конец**

```python
with open("a.txt", "a", encoding="utf-8") as f:
    f.write("more\n")
```

- **Записать список строк**

```python
lines = ["a\n", "b\n", "c\n"]
with open("a.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)
```

- **Чтение и запись одновременно (обновление “на месте” обычно не делают)**  
Обычно читают → меняют → пишут в новый файл/временный, потом заменяют.

### Пути: склейка, текущая папка, абсолютные пути

- **`pathlib` (рекомендуется вместо ручных строк)**

```python
from pathlib import Path

p = Path("data") / "a.txt"
print(p)
print(p.resolve())
```

- **Текущая директория процесса**

```python
from pathlib import Path
print(Path.cwd())
```

### Проверки: файл или папка, существует ли

```python
from pathlib import Path

p = Path("data")
print(p.exists())
print(p.is_dir())
print(p.is_file())
```

---

## Сканирование директорий

### 1) “Только текущий уровень” (файлы/папки рядом)

- **`os.listdir` (имена, без путей)**

```python
import os

names = os.listdir(".")
print(names)
```

- **`Path.iterdir()` (сразу полноценные пути-объекты)**

```python
from pathlib import Path

for entry in Path(".").iterdir():
    print(entry, entry.is_dir(), entry.is_file())
```

- **Отдельно файлы и папки (только текущий уровень)**

```python
from pathlib import Path

root = Path(".")
files = [p for p in root.iterdir() if p.is_file()]
dirs = [p for p in root.iterdir() if p.is_dir()]
print(files)
print(dirs)
```

### 2) “Только текущий уровень, но по маске”

- **Все `.txt` только в этой папке**

```python
from pathlib import Path

for p in Path(".").glob("*.txt"):
    print(p)
```

### 3) “Рекурсивно: папка + все подуровни”

- **Все `.txt` во всех поддиректориях**

```python
from pathlib import Path

for p in Path(".").rglob("*.txt"):
    print(p)
```

- **Все файлы вообще (рекурсивно)**

```python
from pathlib import Path

for p in Path(".").rglob("*"):
    if p.is_file():
        print(p)
```

---

## Как “спускаться в подуровни” (ручной обход)

Идея: на каждом уровне берёшь список подпапок и “заходишь” в каждую.

- **Простой рекурсивный обход**

```python
from pathlib import Path

def walk(dir_path):
    for entry in Path(dir_path).iterdir():
        print(entry)
        if entry.is_dir():
            walk(entry)

walk(".")
```

- **Итеративный обход (стеком), чтобы “спускаться” вручную**

```python
from pathlib import Path

stack = [Path(".")]

while stack:
    d = stack.pop()
    for entry in d.iterdir():
        print(entry)
        if entry.is_dir():
            stack.append(entry)
```

---

## Готовая “карта”: `os.walk` (классика для рекурсии)

`os.walk(root)` на каждом шаге даёт:
- `root`: текущая папка
- `dirs`: список подпапок в ней (имена)
- `files`: список файлов в ней (имена)

```python
import os

for root, dirs, files in os.walk("."):
    print("DIR", root)
    print(" subdirs", dirs)
    print(" files", files)
```

---

## Частые мини-рецепты

- **Создать папку (и не падать, если уже есть)**

```python
from pathlib import Path
Path("out").mkdir(parents=True, exist_ok=True)
```

- **Считать путь как строку для библиотек**

```python
from pathlib import Path
p = Path("a.txt")
print(str(p))
```

- **Имя файла, расширение, родительская папка**

```python
from pathlib import Path
p = Path("dir/file.mp3")
print(p.name)
print(p.suffix)
print(p.parent)
```

---

## Быстрый выбор “что использовать”

- **Чтение/запись**: `open(..., encoding="utf-8")` + `with`
- **Текущий уровень**: `Path.iterdir()`
- **Маска в текущем уровне**: `Path.glob("*.ext")`
- **Рекурсивно по маске**: `Path.rglob("*.ext")`
- **Рекурсивно “папка → подпапки → файлы” с разделением**: `os.walk()`

