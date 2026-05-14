# Librosa

`librosa` — библиотека на Python для анализа аудио (музыки): загрузка, спектральные признаки, темп/бит, MFCC, chroma и другие числовые характеристики сигнала.

Ниже — колонки, которые получаются из “сырых” DSP-метрик (и их агрегатов) и пишутся в `music_data`.

```text
id					- Уникальный id трека в таблице.
game				- Идентификатор игры/источника, часть пути `audios/{game}/{path}`.
path				- Относительный путь к файлу внутри игры, часть пути `audios/{game}/{path}`.
duration_sec		- Длительность аудио в секундах.

rms_mean			- Среднее RMS (энергия/громкость) по окнам.
rms_std				- Стандартное отклонение RMS по окнам.
rms_max				- Максимальное значение RMS по окнам.
rms_p05				- 5-й перцентиль RMS по окнам (уровень “тихих” частей).
rms_p95				- 95-й перцентиль RMS по окнам (уровень “громких” частей).
dyn_range_p95_p05	- Динамический диапазон по RMS: `rms_p95 - rms_p05`.

tempo				- Оценка темпа (BPM) по бит-трекингу.
onset_rate_hz		- Частота атак (onsets) в секунду.
beat_strength_mean	- Средняя “бит-активность” (сила onset envelope/пульса).
beat_strength_std	- Изменчивость “бит-активности” по времени.

spectral_centroid_mean	- Средняя спектральная центроидность (яркость, в Hz).
spectral_centroid_std	- Разброс спектральной центроидности.
spectral_centroid_p05	- 5-й перцентиль спектральной центроидности.
spectral_centroid_p50	- Медиана спектральной центроидности.
spectral_centroid_p95	- 95-й перцентиль спектральной центроидности.

spectral_bandwidth_mean	- Средняя спектральная ширина (bandwidth).
spectral_bandwidth_std		- Разброс спектральной ширины.
spectral_bandwidth_p05		- 5-й перцентиль спектральной ширины.
spectral_bandwidth_p50		- Медиана спектральной ширины.
spectral_bandwidth_p95		- 95-й перцентиль спектральной ширины.

spectral_rolloff_mean	- Средний rolloff (частота, ниже которой заданная доля энергии).
spectral_rolloff_std	- Разброс rolloff по времени.
spectral_rolloff_p05	- 5-й перцентиль rolloff.
spectral_rolloff_p50	- Медиана rolloff.
spectral_rolloff_p95	- 95-й перцентиль rolloff.

spectral_flatness_mean	- Средняя spectral flatness (шумность vs тональность, ближе к 1 = “шумнее”).
spectral_flatness_std	- Разброс spectral flatness.
spectral_flatness_p05	- 5-й перцентиль spectral flatness.
spectral_flatness_p50	- Медиана spectral flatness.
spectral_flatness_p95	- 95-й перцентиль spectral flatness.

spectral_zcr_mean	- Средняя zero-crossing rate (шумность/“резкость” сигнала).
spectral_zcr_std	- Разброс zero-crossing rate.
spectral_zcr_p05	- 5-й перцентиль zero-crossing rate.
spectral_zcr_p50	- Медиана zero-crossing rate.
spectral_zcr_p95	- 95-й перцентиль zero-crossing rate.

mfcc_mean_00..19	- Средние MFCC[0..19] по времени (агрегаты тембра).
mfcc_std_00..19		- Стандартные отклонения MFCC[0..19] по времени (изменчивость тембра).
mfcc_p05_00..19		- 5-е перцентили MFCC[0..19] по времени (нижние уровни коэффициентов).
mfcc_p95_00..19		- 95-е перцентили MFCC[0..19] по времени (верхние уровни коэффициентов).

chroma_entropy		- Энтропия chroma-профиля (насколько тональность “размазана”).
chroma_peakiness	- “Пиковость” chroma (насколько выделяется доминирующая нота).
chroma_mean_00..11	- Средние значения chroma[0..11] по времени (12 классов высоты).

climax_count			- Число обнаруженных “кульминаций” (в БД тип double; при офлайн нормализации считается через ``ln(1+count)``, затем z-score).
climax_density_per_min	- Плотность кульминаций (кол-во в минуту).
climax_strength_mean	- Средняя сила кульминаций.
climax_strength_max		- Максимальная сила кульминаций.
climax_strength_p95		- 95-й перцентиль силы кульминаций.
energy_change_mean		- Средняя величина изменений энергии между окнами.
energy_change_p95		- 95-й перцентиль изменений энергии между окнами.
```

---

# Jamendo

Колонки ниже — вероятности (0..1) “человеко-подобных” тегов MTG-Jamendo, которые считаются поверх эмбеддингов от `discogs-effnet`. Внутри каждого подтипа модель выдаёт вектор по всем классам; в БД эти классы разложены в отдельные колонки с префиксом подтипа.

## Жанры (`jamendo_genre_*`, 87 колонок)

```text
60s 70s 80s 90s acidjazz alternative
alternativerock ambient atmospheric blues bluesrock
bossanova breakbeat celtic chanson chillout
choir classical classicrock club contemporary
country dance darkambient darkwave deephouse
disco downtempo drumnbass dub dubstep
easylistening edm electronic electronica electropop
ethno experimental folk funk fusion
groove grunge hard hardrock hiphop
house idm improvisation indie industrial
instrumentalpop instrumentalrock jazz jazzfusion latin
lounge medieval metal minimal newage
newwave pop popfolk poprock postrock
progressive psychedelic punkrock rap reggae
rnb rock rocknroll singersongwriter soul
soundtrack swing symphonic synthpop techno
trance triphop world worldfusion
```

## Инструменты (`jamendo_instrument_*`, 40 колонок)

```text
accordion acousticbassguitar acousticguitar bass
beat bell bongo brass cello
clarinet classicalguitar computer doublebass
drummachine drums electricguitar electricpiano
flute guitar harmonica harp horn
keyboard oboe orchestra organ pad
percussion piano pipeorgan rhodes sampler
saxophone strings synthesizer trombone
trumpet viola violin voice
```

## Настроения / Темы (`jamendo_moodtheme_*`, 56 колонок)

```text
action adventure advertising background
ballad calm children christmas
commercial cool corporate dark
deep documentary drama dramatic
dream emotional energetic epic
fast film fun funny game
groovy happy heavy holiday
hopeful inspiring love melancholic
melodic meditative motivational movie
nature party positive powerful
relaxing retro romantic sad
sexy slow soft soundscape
space sport summer trailer
travel upbeat uplifting
```

---

# Embeddings

Эмбеддинги — это “сжатое представление” аудио, которое извлекается нейросетевой моделью (сейчас внутри `discogs-effnet` для essentia моделей) и используется как вход для более “человеко-подобных” классификаторов (Jamendo жанры/инструменты/настроения).

Этот пункт не входит в таблицу, но тут важно упомянуть, что помимо “низкоуровневых” метрик (RMS/темп/спектр/MFCC/chroma/пики), можно пробовать подавать **готовые нейросетевые эмбеддинги** и/или их агрегаты как высокоуровневый сигнал: они часто лучше выражают сложные свойства типа настроения/жанра/инструментов, чем набор отдельных DSP-признаков.

