DB ?= MIMICIV_20

DATA_RAW_PATH ?= $(abspath ./data/raw)

MIMICIV_20_NAME ?= mimic-iv-2.0
MIMICIV_20_URL ?= https://physionet.org/files/mimiciv/2.0/
MIMICIV_20_PATH ?= $(DATA_RAW_PATH)/physionet.org/files/mimiciv/2.0

DB_NAME = ${${DB}_NAME}
DB_URL = ${${DB}_URL}
DB_PATH = ${${DB}_PATH}

PHONY += physionet-download
physionet-download:
ifndef PHYSIONET_USER
	$(error PHYSIONET_USER is undefined)
endif
ifndef PHYSIONET_PASSWORD
	$(error PHYSIONET_PASSWORD is undefined)
endif
	wget -r -N -c -np --user $(PHYSIONET_USER) --password $(PHYSIONET_PASSWORD) -P $(DATA_RAW_PATH) $(DB_URL)
