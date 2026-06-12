-- Create a function that always returns the first non-NULL value:
CREATE OR REPLACE FUNCTION public.first_agg (anyelement, anyelement)
  RETURNS anyelement
  LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
'SELECT $1';

-- Then wrap an aggregate around it:
CREATE OR REPLACE AGGREGATE public.first (anyelement) (
  SFUNC    = public.first_agg
, STYPE    = anyelement
, PARALLEL = safe
);

-- Create a function that always returns the last non-NULL value:
CREATE OR REPLACE FUNCTION public.last_agg (anyelement, anyelement)
  RETURNS anyelement
  LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
'SELECT $2';

-- Then wrap an aggregate around it:
CREATE OR REPLACE AGGREGATE public.last (anyelement) (
  SFUNC    = public.last_agg
, STYPE    = anyelement
, PARALLEL = safe
);

DROP TABLE IF EXISTS public.icd10_to_icd9;
CREATE TABLE public.icd10_to_icd9
(
    icd10_code CHAR(7) NOT NULL,
    icd9_code CHAR(7) NOT NULL,
    flag CHAR(5) NOT NULL
);


DROP TABLE IF EXISTS public.icd9_to_ccs;
CREATE TABLE public.icd9_to_ccs
(
    icd9_code CHAR(7) NOT NULL,
    ccs INT NOT NULL,
    ccs_label VARCHAR(64) NOT NULL
);


DROP TABLE IF EXISTS public.icd_to_phecode;
CREATE TABLE public.icd_to_phecode
(
    icd CHAR(16) NOT NULL,
    flag INT NOT NULL,
    phecode CHAR(16) NOT NULL
);
