#!/bin/bash

psql "dbname=mimiciv_20 user='$POSTGRES_USER'" < /data/concepts/postgres-functions.sql
psql "dbname=mimiciv_20 user='$POSTGRES_USER'" < /data/concepts/postgres-make-concepts.sql
psql "dbname=mimiciv_20 user='$POSTGRES_USER'" < /data/concepts/index.sql

pg_ctl stop
