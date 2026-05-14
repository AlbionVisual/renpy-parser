\q	Quit/Exit psql.
\l	List all databases.
\c [dbname]	Connect to a specific database.
\dt	List all tables in the current schema.
\dt *.*	List all tables in all schemas.
\d [table_name]	Describe a table (columns, types, constraints, indexes).
\d+ [table_name]	Show more detailed table information, including size and description.
\dn	List all schemas.
\df	List all functions.
\du	List all users and their roles.
\di	List all indexes.
\dv	List all views.
\x [on|off|auto]	Toggle expanded output display (useful for wide tables).
\timing [on|off]	Toggle display of query execution time.
\i [filename]	Execute SQL commands from a file.
\o [filename]	Send all query results to a file (until another \o or \q).
\copy ...	Perform client-side data import/export to/from a file (e.g., CSV).
\!	Execute a shell command.
\?	Show help for all psql backslash commands.