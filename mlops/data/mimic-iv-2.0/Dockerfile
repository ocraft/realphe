FROM postgres:latest

ENV DATA_TO_LOAD=/data_to_load
ENV DATA_EXTERNAL=/data_external

RUN mkdir -p ${DATA_TO_LOAD} /data/setup /data/concepts /data/extend ${DATA_EXTERNAL}
COPY ./setup /data/setup
COPY ./concepts /data/concepts
COPY ./extend /data/extend
RUN chmod +x /data/setup/setup.sh
RUN chmod +x /data/concepts/setup.sh
RUN chmod +x /data/extend/setup.sh

EXPOSE 5432
VOLUME ${DATA_TO_LOAD}
VOLUME ${DATA_EXTERNAL}
