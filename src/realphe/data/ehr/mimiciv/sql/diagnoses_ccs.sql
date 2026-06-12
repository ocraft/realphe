WITH dx AS (
	SELECT
        icu.stay_id,
        MIN(d.seq_num) AS seq_num,
        ccs_map.ccs,
        MAX(ccs_map.ccs_label) AS ccs_label
    FROM mimiciv_hosp.diagnoses_icd d
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = d.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id = icu.stay_id
    LEFT JOIN public.icd10_to_icd9 icd_map ON D.icd_version = 10 AND icd_map.icd10_code = d.icd_code
    LEFT JOIN public.icd9_to_ccs ccs_map
    ON COALESCE(icd_map.icd9_code, d.icd_code) = ccs_map.icd9_code
    GROUP BY icu.stay_id, ccs
    ORDER BY icu.stay_id, seq_num, ccs
),
dx_cnt AS (
 	SELECT
        ccs,
		COUNT(ccs) cnt
    FROM dx
	GROUP BY dx.ccs
)
SELECT
	dx.stay_id,
    row_number() OVER(PARTITION BY dx.stay_id ORDER BY dx.seq_num) dx_seq_num,
	dx.ccs AS ccs,
	dx.ccs_label AS ccs_name
FROM dx
JOIN dx_cnt c ON dx.ccs = c.ccs AND c.cnt >= {diagnoses_min_samples}
WHERE dx.ccs IS NOT NULL
GROUP BY dx.stay_id, dx.seq_num, dx.ccs, dx.ccs_label
ORDER BY dx.stay_id, dx.seq_num, dx.ccs
