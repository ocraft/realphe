WITH dx AS (
    SELECT
        icu.stay_id,
        MIN(d.seq_num) AS seq_num,
        COALESCE(icd_map.icd9_code, d.icd_code) AS icd9
    FROM mimiciv_hosp.diagnoses_icd d
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = d.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id = icu.stay_id
    LEFT JOIN public.icd10_to_icd9 icd_map ON d.icd_version = 10 AND icd_map.icd10_code = d.icd_code
    GROUP BY icu.stay_id, icd9
    ORDER BY icu.stay_id, seq_num, icd9
),
dx_cnt AS (
 	SELECT
        icd9,
		COUNT(icd9) cnt
    FROM dx
	GROUP BY dx.icd9
)
SELECT
	dx.stay_id,
    row_number() OVER(PARTITION BY dx.stay_id ORDER BY dx.seq_num) dx_seq_num,
	TRIM(dx.icd9) AS icd9,
	dx_info.long_title AS icd9_name
FROM dx
JOIN dx_cnt c ON dx.icd9 = c.icd9 AND c.cnt >= {diagnoses_min_samples}
JOIN mimiciv_hosp.d_icd_diagnoses dx_info ON dx_info.icd_code=dx.icd9 AND dx_info.icd_version=9
GROUP BY dx.stay_id, dx.seq_num, dx.icd9, dx_info.long_title
ORDER BY dx.stay_id, dx.seq_num, dx.icd9
