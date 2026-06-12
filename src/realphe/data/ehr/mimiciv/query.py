from collections.abc import Iterator
from enum import Enum
from importlib.resources import read_text
from typing import Any, Dict, Optional

import pandas as pd
from pandas.core.generic import DtypeArg
import sqlalchemy
from sqlalchemy.sql.elements import TextClause

from . import sql


class MimicSql(Enum):
    COHORT = 'cohort.sql'
    TIME_SERIES_STATE = 'time_series_state.sql'
    TIME_SERIES_STATE_COUNT = 'time_series_state_count.sql'
    DIAGNOSES_ICD9 = 'diagnoses_icd9.sql'
    DIAGNOSES_CCS = 'diagnoses_ccs.sql'
    DIAGNOSES_PHECODE = 'diagnoses_phecode.sql'


def mimic_query(mimic_sql: MimicSql, params: Optional[Dict[str, Any]] = None) -> TextClause:
    query = read_text(sql, mimic_sql.value)
    if params is not None:
        query = query.format(**params)
    return sqlalchemy.text(query)


class MimicDb:
    def __init__(
        self,
        host='localhost',
        dbname='realphe_data',
        user='postgres',
        password='postgres'
    ):
        engine = sqlalchemy.create_engine(
            f'postgresql+psycopg2://{user}:{password}@{host}/{dbname}',
            connect_args={'options': '-csearch_path=mimiciv_hosp,mimiciv_icu,mimiciv_derived'},
            client_encoding="utf8")
        self.con = engine.connect().execution_options(stream_results=True)

    def execute(self,
                query: MimicSql,
                criteria: Optional[Dict[str, Any]] = None,
                dtype: Optional[DtypeArg] = None,
                chunksize: Optional[int] = None) -> pd.DataFrame:
        if chunksize is None:
            return pd.read_sql_query(mimic_query(query, criteria), self.con, dtype=dtype)
        df_final = pd.DataFrame()
        for chunk in pd.read_sql_query(mimic_query(query, criteria),
                                       self.con,
                                       dtype=dtype,
                                       chunksize=chunksize):
            df_final = pd.concat([df_final, chunk])
        return df_final.astype(dtype)

    def query(self,
              sql_query: str,
              chunksize: Optional[int] = None) -> pd.DataFrame | Iterator[pd.DataFrame]:
        return pd.read_sql_query(sql_query, self.con, chunksize=chunksize)
