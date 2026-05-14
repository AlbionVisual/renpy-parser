Что вообще у нас происходит с текстом:
```sql
select game, phrase_order, music, text_translated from text_musics_rel where music is not null and text_translated is not null order by game, phrase_order;
```

Странные строки:
```sql
select
  game,
  phrase_order,
  text1
from text_musics_rel
where text1 is not null
  and (
    text1 ~ '\.{4,}'
    or text1 ~ '\s{3,}'
    or text1 ~ '\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\]'
    or text1 ~ '\{[^}]*\}'
    or text1 ~ '<[^<>]+>'
    or text1 ~ '\\\\[nrt\"\\\\]'
    or text1 !~ '^[А-Яа-яЁё0-9[:space:]''"_\.\,\!\?\;\:\-\—–…«»“”„()/%№]+$'
  )
order by game, phrase_order;
```

Составь свой запрос:
```sql
select game, count(*) from text_musics_rel
where text1 is not null and (
text1 ~ '\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\]'
) group by game order by game;

or text1 ~ '\.{4,}'
or text1 ~ '\s{2,}'
or text1 ~ '\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\]'
or text1 ~ '\{[^}]*\}'
or text1 ~ '\\\\[nrt\"\\\\]'
or text1 !~ '^[А-Яа-яЁё0-9[:space:]''"_\.\,\!\?\;\:\-\—–…«»“”„()/%№]+$'

select game, phrase_order, text1 from text_musics_rel
where text1 is not null and (
text1 ~ '\.{4,}'
or text1 ~ '\s{2,}'
or text1 ~ '\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\]'
or text1 ~ '\{[^}]*\}'
or text1 ~ '\\\\[nrt\"\\\\]'
or text1 !~ '^[А-Яа-яЁё0-9[:space:]''"_\.\,\!\?\;\:\-\—–…«»“”„()/%№]+$'
  ) order by game, phrase_order;
```

### Таблица `text_musics_rubert` (RuBERT)

Таблица есть, строки скопированы из `text_musics_rel`:

```sql
select count(*) from text_musics_rubert;
```

Сколько строк с текстом и сколько уже с эмбеддингом:

```sql
select
  count(*) filter (where text1 is not null and btrim(text1) <> '') as with_text1,
  count(*) filter (where text_analized is not null) as with_embedding
from text_musics_rubert;
```

Проверка размерности вектора (у `DeepPavlov/rubert-base-cased` hidden = 768):

```sql
select
  game,
  phrase_order,
  cardinality(text_analized) as dim
from text_musics_rubert
where text_analized is not null
limit 20;
```

Строки без эмбеддинга при непустом `text1` (ещё не прогнали encode):

```sql
select game, phrase_order, left(text1, 80) as snippet
from text_musics_rubert
where text1 is not null and btrim(text1) <> '' and text_analized is null
order by game, phrase_order
limit 50;
```

Случайный sanity-check значений первых координат:

```sql
select game, phrase_order, text_analized[1:4] as head4
from text_musics_rubert
where text_analized is not null
order by random()
limit 5;
```

Статистика по **компонентам** вектора `text_analized` (пер-измерение: среднее и СКО по всем строкам; удобно для z-score в SQL/WHERE):

```sql
select
  ordinality AS dim_ix,
  avg(val::double precision) AS mean,
  stddev_pop(val::double precision) AS stddev_pop,
  stddev_samp(val::double precision) AS stddev_samp
from text_musics_rubert
cross join lateral unnest(text_analized)
  with ordinality as t(val, ordinality)
where text_analized is not null
  and cardinality(text_analized) > 0
group by ordinality
order by ordinality;
```

Сводка по **L2-норме** целого вектора (одно число на строку — «насколько длинный» эмбеддинг):

```sql
with norms as (
  select
    sqrt(
      coalesce(
        (select sum((x)::double precision * (x)::double precision)
         from unnest(text_analized) as x),
        0.0
      )
    ) as l2
  from text_musics_rubert
  where text_analized is not null
    and cardinality(text_analized) > 0
)
select
  avg(l2) as mean_l2,
  stddev_pop(l2) as stddev_pop_l2,
  stddev_samp(l2) as stddev_samp_l2,
  count(*) as n_rows
from norms;
```

### Таблица `music_data` (аудио-фичи)

Сводка строк и ключей:

```sql
select count(*) from music_data;
```

Несколько строк с основными колонками (подставьте свои имена колонок при необходимости):

```sql
select game, path, duration_sec, tempo, rms_mean, climax_count
from music_data
order by game, path
limit 20;
```

Одна строка целиком (удобно в `psql` с `\x`):

```sql
select * from music_data
where game = 'SmolBirb-pc' and path = 'audio/ambient4blues.mp3';
```

Метрики нормализации (mean / stddev / эффективный знаменатель после `finalize_training_normalization`):

```sql
select target_table, processor_key, column_name, mean, stddev
from audio_processor_norm_metrics
where target_table = 'music_data'
order by processor_key, column_name;
```

Проверка, что по треку заполнены DSP-колонки (пример — spectral):

```sql
select game, path, spectral_centroid_mean, mfcc_mean_00
from music_data
where spectral_centroid_mean is not null
limit 10;
```