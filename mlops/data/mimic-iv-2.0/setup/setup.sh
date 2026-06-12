#!/bin/bash

echo 'Preparing MIMIC-IV 2.0 ... '

echo "- running create 'mimiciv_20' db if not exists ..."

psql <<- EOSQL
SELECT 'CREATE DATABASE mimiciv_20 OWNER postgres' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mimiciv_20')\gexec
\c mimiciv_20;
EOSQL

echo "- health checks ..."
# check for the admissions to set the extension
if [ -e "/data_to_load/hosp/admissions.csv.gz" ]; then
  COMPRESSED=1
  EXT='.csv.gz'
elif [ -e "/data_to_load/hosp/admissions.csv" ]; then
  COMPRESSED=0
  EXT='.csv'
else
  echo "Error: Unable to find a MIMIC-IV 2.0 data file (admissions) in /data_to_load/hosp"
  echo "Error: Did you map a local directory using `docker run -v /path/to/mimic/data:/data_to_load` ?"
  exit 1
fi

# check for all the tables, exit if we are missing any
ALLTABLES_HOSP='admissions d_hcpcs d_icd_diagnoses d_icd_procedures d_labitems diagnoses_icd drgcodes emar emar_detail hcpcsevents labevents microbiologyevents omr patients pharmacy poe poe_detail prescriptions procedures_icd services transfers'
ALLTABLES_ICU='chartevents d_items datetimeevents icustays ingredientevents inputevents outputevents procedureevents'

for TBL in $ALLTABLES_HOSP; do
  if [ ! -e "/data_to_load/hosp/${TBL}$EXT" ];
  then
    echo "Unable to find ${TBL}$EXT in /data_to_load/hosp"
    exit 1
  fi
  echo "Found all tables in /data_to_load/hosp - beginning import from $EXT files."
done

for TBL in $ALLTABLES_ICU; do
  if [ ! -e "/data_to_load/icu/${TBL}$EXT" ];
  then
    echo "Unable to find ${TBL}$EXT in /data_to_load/icu"
    exit 1
  fi
  echo "Found all tables in /data_to_load/icu - beginning import from $EXT files."
done

# checks passed - begin building the database
echo "- $0: running create.sql"
psql "dbname=mimiciv_20 user='$POSTGRES_USER'" < /data/setup/create.sql

if [ $COMPRESSED -eq 1 ]; then
  echo "- $0: running load_gz.sql"
  psql "dbname=mimiciv_20 user='$POSTGRES_USER'" -v mimic_data_dir=/data_to_load < /data/setup/load_gz.sql
else
  echo "- $0: running load.sql"
  psql "dbname=mimiciv_20 user='$POSTGRES_USER'" -v mimic_data_dir=/data_to_load < /data/setup/load.sql
fi

echo "- $0: running constraint.sql"
psql "dbname=mimiciv_20 user='$POSTGRES_USER'" < /data/setup/constraint.sql

echo "- $0: running index.sql"
psql "dbname=mimiciv_20 user='$POSTGRES_USER'" < /data/setup/index.sql

echo "- $0: running validate.sql (all rows should return PASSED)"
psql "dbname=mimiciv_20 user='$POSTGRES_USER'" < /data/setup/validate.sql

echo '... Done!'
