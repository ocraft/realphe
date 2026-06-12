#!/bin/bash

echo "$0: running create.sql"
psql "dbname=mimiciv_20 user='$POSTGRES_USER' options=--search_path=public" < /data/extend/create.sql

echo "$0: running load.sql"
psql -1 "dbname=mimiciv_20 user='$POSTGRES_USER' options=--search_path=public" -v mimic_data_dir=/data_external < /data/extend/load.sql

echo "$0: running ext.sql"
psql "dbname=mimiciv_20 user='$POSTGRES_USER' options=--search_path=public" < /data/extend/ext.sql

echo "$0: running charts.sql"
psql "dbname=mimiciv_20 user='$POSTGRES_USER' options=--search_path=public" < /data/extend/charts.sql

echo "$0: running index.sql"
psql "dbname=mimiciv_20 user='$POSTGRES_USER' options=--search_path=public" < /data/extend/index.sql
