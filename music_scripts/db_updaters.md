Select empty
```sql
select DISTINCT ON (game, music_path) game, music_path, raw_music_line, music from text_musics_rel where music is null and music_path != 'NO_MUSIC' order by game, music_path, line_number;
```
Count empty
```sql
select count(*) from text_musics_rel where music is null and music_path != 'NO_MUSIC';
```
Update sth
```sql
update text_musics_rel set music_path = 'sound/SOSPetlya.ogg' where game = 'MoeEra-win' and music_path = 'UNKNOWN:<silence 4.0>';
```
Remove unused words from names
```sql
update text_musics_rel t set music = m.id from music_data m where t.game = 'MilgramSubstitution-DEMO-1.4.2-pc' and t.music_path like '%UNKNOWN:%' and t.music is null and(
     lower(trim(regexp_replace(t.music_path, '^.*UNKNOWN:', ''))) =
        trim(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(lower(path), '^.*/', ''),
                        '\.[a-z0-9]+$', ''
                    ),
                    '( - suno ai| - udio ai)$', ''
                ),
                ' ', '', 'g'
            )
        )
);
```
Try regexps
```sql
select trim(regexp_replace(music_path, '^.*UNKNOWN:', '')), music_path, from text_musics_rel where game = 'MilgramSubstitution-DEMO-1.4.2-pc';
```
```sql
select path, trim(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(path, '^.*/', ''),
                        '\.[a-zA-Z0-9]+$', ''
                    ),
                    '( - Suno Ai| - Udio Ai)$', ''
                ),
                ' ', '', 'g'
            )
        ) from music_data where game = 'MilgramSubstitution-DEMO-1.4.2-pc';
```
Fasten join in update 
```sql
create index if not exists idx_tmr_music_null on text_musics_rel(game) where music is null;
create index if not exists idx_md_game on music_data(game);
```