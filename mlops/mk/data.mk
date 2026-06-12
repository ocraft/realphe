VOLUME ?= realphe_data

DATA_EXTERNAL_PATH ?= $(abspath ./data/external)

# args: db,data_to_load,working_dir to call setup.sh
define postgres_setup
	docker build -t postgres/$1 ./mlops/data/$1/
	docker volume create --name $(VOLUME)
	docker run --name $1 \
		-e POSTGRES_PASSWORD=postgres \
		-v $2:/data_to_load \
		-v $(VOLUME):/var/lib/postgresql/data \
		-v $(DATA_EXTERNAL_PATH):/data_external \
		-d \
		postgres/$1 \
		postgres \
		-c checkpoint_timeout=600 \
		-c max_wal_size=4GB \
		-c shared_preload_libraries='pg_stat_statements'

	docker exec --user postgres $1 bash -c '\
		for i in {1..30}; do \
			pg_isready -U postgres && exit 0; \
			sleep 1; \
		done; \
		echo "PostgreSQL not ready in time" >&2; \
		exit 1'

	docker exec --user postgres -w $3 $1 ./setup.sh
	docker stop -t 120 $1
	docker container rm $1
	docker image rm postgres/$1
endef

PHONY += data-setup
data-setup:
	$(call postgres_setup,$(DB_NAME),$(DB_PATH),/data/setup)

PHONY += data-concepts
data-concepts:
	$(call postgres_setup,$(DB_NAME),$(DB_PATH),/data/concepts)

PHONY += data-extend
data-extend:
	$(call postgres_setup,$(DB_NAME),$(DB_PATH),/data/extend)
