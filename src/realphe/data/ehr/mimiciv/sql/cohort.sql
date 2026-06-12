WITH
serv AS (
    WITH
    serv_rank AS
    (
        SELECT
            icu.hadm_id,
            icu.stay_id,
            ser.curr_service,
            CASE
                WHEN curr_service LIKE '%SURG' THEN 1
                WHEN curr_service = 'ORTHO' THEN 1
                ELSE 0
            END AS surgical,
            RANK() OVER (PARTITION BY icu.hadm_id ORDER BY ser.transfertime DESC) as services_seq
        FROM mimiciv_icu.icustays icu
        LEFT JOIN mimiciv_hosp.services ser ON icu.hadm_id = ser.hadm_id AND
        ser.transfertime < icu.intime + INTERVAL '12' HOUR
    ),
    serv_cnt AS
    (
        SELECT
            icu.stay_id,
            COUNT(ser.curr_service) cnt
        FROM mimiciv_icu.icustays icu
        LEFT JOIN mimiciv_hosp.services ser ON icu.hadm_id = ser.hadm_id
        AND ser.transfertime < icu.intime + INTERVAL '12' HOUR
        GROUP BY icu.stay_id
    )
    SELECT
        serv_rank.hadm_id,
        serv_rank.stay_id,
        serv_rank.curr_service,
        serv_rank.surgical,
        CASE
            WHEN serv_cnt.cnt > 1 THEN 1
            ELSE 0
        END AS many_services
    FROM serv_rank
    LEFT JOIN serv_cnt ON serv_cnt.stay_id = serv_rank.stay_id
    WHERE services_seq = 1
),
co AS (
    SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.stay_id,
        icu.intime,
        icu.outtime,
        icu.first_careunit,
        adm.race,
        adm.insurance,
        adm.marital_status,
        adm.admission_type,
        adm.admission_location,
        adm.admittime,
        adm.language,
        -- in-hospital (only!) time of death
        adm.deathtime,
        adm.hospital_expire_flag,
        icu_times.intime_hr,
        icu_times.outtime_hr,
        icu_d.gender,
        -- first ICU stay *for the current hospitalization*
        icu_d.first_icu_stay,
        icu_d.first_hosp_stay,
        icu_d.los_icu * 24 AS los_icu_hours,
        icu_d.los_hospital * 24 AS los_hospital_hours,
        icu_d.icustay_seq,
        icu_d.hospstay_seq,
        -- if > 89 capped to 91;
        icu_d.admission_age AS age,
        CASE
            WHEN icu_d.dod > icu.outtime AND icu_d.dod <= icu.outtime + interval '30' DAY THEN 1
            ELSE 0
        END AS thirtyday_expire_flag,
        CASE
            WHEN adm.deathtime IS NULL THEN 0
            WHEN adm.deathtime <= icu.outtime THEN 1
            ELSE 0
        END AS icu_expire_flag
    FROM mimiciv_icu.icustays icu
    LEFT JOIN mimiciv_hosp.admissions adm ON icu.hadm_id = adm.hadm_id
    LEFT JOIN mimiciv_derived.icustay_detail icu_d ON icu_d.stay_id = icu.stay_id
    LEFT JOIN mimiciv_derived.icustay_times icu_times ON icu_times.stay_id = icu.stay_id
),
his AS (
    SELECT
        his.stay_id,
        MAX(his.iv_access_prior_to_admission) AS iv_access_prior_to_admission,
        MAX(his.insulin_pump) AS insulin_pump,
        MAX(his.self_activities_of_daily_living) AS self_activities_of_daily_living,
        MAX(his.history_of_falls) AS history_of_falls,
        MAX(his.home_tube_feeding) AS home_tube_feeding,
        MAX(his.ethyl_alcohol) AS ethyl_alcohol,
        MAX(his.pressure_ulcer_present) AS pressure_ulcer_present,
        MAX(his.difficulty_swallowing) AS difficulty_swallowing,
        MAX(his.visual_hearing_deficit) AS visual_hearing_deficit,
        MAX(his.currently_experiencing_pain) AS currently_experiencing_pain,
        MAX(his.dialysis_patient) AS dialysis_patient
    FROM mimiciv_derived.charts his
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = his.stay_id
    GROUP BY his.stay_id
),
htm AS (
    SELECT
        htm.stay_id,
        public.LAST(htm.height) height
    FROM mimiciv_derived.height htm
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = htm.stay_id
    GROUP BY htm.stay_id
)
SELECT
    co.subject_id,
    co.hadm_id,
    co.stay_id,
    co.icustay_seq,
    co.hospstay_seq,
    co.age,
    co.gender,
    co.race,
    co.first_careunit,
    co.insurance,
    co.marital_status,
    co.admission_type,
    co.admission_location,
    co.admittime AS admission_time,
    co.language,
    serv.curr_service AS last_service,
    co.intime_hr,
    co.outtime_hr,
    htm.height,

    -- admission history
    his.iv_access_prior_to_admission,
    his.insulin_pump,
    his.self_activities_of_daily_living,
    his.history_of_falls,
    his.home_tube_feeding,
    his.ethyl_alcohol,
    his.pressure_ulcer_present,
    his.difficulty_swallowing,
    his.visual_hearing_deficit,
    his.currently_experiencing_pain,
    his.dialysis_patient,

    -- outcomes
    co.hospital_expire_flag,
    co.thirtyday_expire_flag,
    co.icu_expire_flag,
    co.los_icu_hours,
    co.los_hospital_hours,
    CEIL(EXTRACT(epoch FROM (co.deathtime - co.intime_hr))/60.0/60.0) AS deathtime_hours,

    --exclusions
    CASE WHEN co.los_icu_hours < {short_icu_los_hours} THEN 1 ELSE 0 END AS exclusion_short_icu_los,
    CASE WHEN co.los_icu_hours > {long_icu_los_hours} THEN 1 ELSE 0 END AS exclusion_long_icu_los,
    CASE WHEN co.age < {adult_age} THEN 1 ELSE 0 END AS exclusion_not_adult,
    CASE WHEN co.first_icu_stay THEN 0 ELSE 1 END AS exclusion_not_first_icu_stay,
    CASE WHEN co.first_hosp_stay THEN 0 ELSE 1 END AS exclusion_not_first_hosp_stay,
    CASE WHEN serv.surgical = 1 THEN 1 ELSE 0 END AS exclusion_surgical,
    CASE WHEN serv.many_services = 1 THEN 1 ELSE 0 END AS exclusion_many_services,
    CASE
        WHEN co.intime IS NULL THEN 1
        WHEN co.outtime IS NULL THEN 1
        WHEN co.intime_hr IS NULL THEN 1
        WHEN co.outtime_hr IS NULL THEN 1
        ELSE 0
    END AS exclusion_invalid_data
FROM co
LEFT JOIN serv ON co.stay_id = serv.stay_id
LEFT JOIN his ON co.stay_id = his.stay_id
LEFT JOIN htm ON co.stay_id = htm.stay_id
