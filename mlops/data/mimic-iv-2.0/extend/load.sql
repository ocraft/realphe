\cd :mimic_data_dir

SET CLIENT_ENCODING TO 'utf8';

CREATE TEMP TABLE icd10_to_icd9_temp ON COMMIT DROP AS
SELECT * FROM public.icd10_to_icd9 WITH NO DATA;

\cd clinical_coding/diagnosis_gems_2018
\COPY icd10_to_icd9_temp FROM 2018_I10gem.csv DELIMITER E'\t' CSV HEADER NULL '';

\cd ../diagnosis_gems_2017
\COPY icd10_to_icd9_temp FROM 2017_I10gem.csv DELIMITER E'\t' CSV HEADER NULL '';

\cd ../diagnosis_gems_2016
\COPY icd10_to_icd9_temp FROM 2016_I10gem.csv DELIMITER E'\t' CSV HEADER NULL '';

\cd ../diagnosis_gems_2015
\COPY icd10_to_icd9_temp FROM 2015_I10gem.csv DELIMITER E'\t' CSV HEADER NULL '';

\cd ../diagnosis_gems_2014
\COPY icd10_to_icd9_temp FROM 2014_I10gem.csv DELIMITER E'\t' CSV HEADER NULL '';

INSERT INTO public.icd10_to_icd9 SELECT DISTINCT * FROM icd10_to_icd9_temp;

\cd ../

CREATE TEMP TABLE icd10_update (
    icd10_code VARCHAR(64),
    year INT,
    icd10_code_prev VARCHAR(64)
) ON COMMIT DROP;

\COPY icd10_update FROM icd10_update_2019.csv DELIMITER ',' CSV HEADER NULL '';
\COPY icd10_update FROM icd10_update_2020.csv DELIMITER ',' CSV HEADER NULL '';

INSERT INTO public.icd10_to_icd9
SELECT DISTINCT n.icd10_code, o.icd9_code, o.flag
FROM icd10_update n
JOIN public.icd10_to_icd9 o ON o.icd10_code = n.icd10_code_prev
WHERE LENGTH(n.icd10_code_prev) < 8 AND LENGTH(n.icd10_code) < 8;

CREATE TEMP TABLE ccs_temp (
    t0 VARCHAR(64),
    t1 VARCHAR(64),
    t2 VARCHAR(64),
    t3 VARCHAR(64),
    t4 VARCHAR(64),
    t5 VARCHAR(64)
) ON COMMIT DROP;

\cd Single_Level_CCS_2015
\COPY ccs_temp FROM '$dxref 2015.csv' DELIMITER ',' CSV HEADER NULL '';

INSERT INTO public.icd9_to_ccs(icd9_code, ccs, ccs_label)
SELECT t0, t1::INT, t2
FROM ccs_temp;

\cd ../
\COPY icd_to_phecode FROM ICD-CM_to_phecode_unrolled.txt DELIMITER E'\t' CSV HEADER;
