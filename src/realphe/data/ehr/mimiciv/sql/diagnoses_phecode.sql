WITH dx AS (
	SELECT
        icu.stay_id,
        MIN(d.seq_num) AS seq_num,
        p.phecode
    FROM mimiciv_hosp.diagnoses_icd d
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = d.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id = icu.stay_id
    LEFT JOIN public.icd10_to_icd9 icd_map ON d.icd_version = 10 AND icd_map.icd10_code = d.icd_code
    LEFT JOIN public.icd_to_phecode p ON p.icd = d.icd_code and p.flag = d.icd_version
    GROUP BY icu.stay_id, phecode
    ORDER BY icu.stay_id, seq_num, phecode
),
dx_cnt AS (
 	SELECT
        phecode,
		COUNT(phecode) cnt
    FROM dx
	GROUP BY dx.phecode
)
SELECT
	dx.stay_id,
    row_number() OVER(PARTITION BY dx.stay_id ORDER BY dx.seq_num) dx_seq_num,
	TRIM(dx.phecode) AS phecode
FROM dx
JOIN dx_cnt c ON dx.phecode = c.phecode AND c.cnt >= {diagnoses_min_samples}
WHERE dx.phecode IS NOT NULL
GROUP BY dx.stay_id, dx.seq_num, dx.phecode
ORDER BY dx.stay_id, dx.seq_num, phecode
