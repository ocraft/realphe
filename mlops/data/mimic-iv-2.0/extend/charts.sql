DROP TABLE IF EXISTS mimiciv_derived.charts; CREATE TABLE mimiciv_derived.charts AS
SELECT
    che.subject_id,
    che.stay_id,
    che.charttime,
    MAX(
        CASE WHEN itemid=223951 AND value='Abnormal >3 Seconds' THEN 1 ELSE 0 END
    ) AS capillary_refill_rate_r,
    MAX(
        CASE WHEN itemid=224308 AND value='Abnormal >3 Seconds' THEN 1 ELSE 0 END
    ) AS capillary_refill_rate_l,
    AVG(
        CASE
        WHEN itemid IN (220274, 223830) AND valuenum>6 AND valuenum<10
            THEN valuenum
        ELSE NULL END
    ) AS ph_blood,
    AVG(
        CASE
        WHEN itemid IN (220734) AND valuenum>3 AND valuenum<10
            THEN valuenum
        ELSE NULL END
    ) AS ph_urine,
    public.LAST(
        CASE WHEN itemid=224641 THEN valuenum ELSE NULL END
    ) AS alarms_on,
    public.LAST(
        CASE WHEN itemid=224168 THEN valuenum ELSE NULL END
    ) AS parameters_checked,
    public.LAST(
        CASE WHEN itemid=220047 AND valuenum>0 AND valuenum<300 THEN valuenum ELSE NULL END
    ) AS heart_rate_alarm_low,
    public.LAST(
        CASE WHEN itemid=220046 AND valuenum>0 AND valuenum<300 THEN valuenum ELSE NULL END
    ) AS heart_rate_alarm_high,
    public.LAST(
        CASE WHEN itemid=223770 AND valuenum>0 AND valuenum<=100 THEN valuenum ELSE NULL END
    ) AS o2_saturation_alarm_low,
    public.LAST(
        CASE WHEN itemid=223769 AND valuenum>0 AND valuenum<=100 THEN valuenum ELSE NULL END
    ) AS o2_saturation_alarm_high,
    public.LAST(
        CASE WHEN itemid=224162 AND valuenum>0 AND valuenum<=70 THEN valuenum ELSE NULL END
    ) AS respiratory_rate_alarm_low,
    public.LAST(
        CASE WHEN itemid=224161 AND valuenum>0 AND valuenum<=70 THEN valuenum ELSE NULL END
    ) AS respiratory_rate_alarm_high,
    public.LAST(
        CASE WHEN itemid=224054 THEN value ELSE NULL END
    ) AS braden_sensory_perception,
    public.LAST(
        CASE WHEN itemid=224057 THEN value ELSE NULL END
    ) AS braden_mobility,
    public.LAST(
        CASE WHEN itemid=224055 THEN value ELSE NULL END
    ) AS braden_moisture,
    public.LAST(
        CASE WHEN itemid=224056 THEN value ELSE NULL END
    ) AS braden_activity,
    public.LAST(
        CASE WHEN itemid=224058 THEN value ELSE NULL END
    ) AS braden_nutrition,
    public.LAST(
        CASE WHEN itemid=224059 THEN value ELSE NULL END
    ) AS braden_friction,
    public.LAST(
        CASE WHEN itemid=226253 AND valuenum>0 AND valuenum<=100 THEN valuenum ELSE NULL END
    ) AS o2_saturation_desat_limit,
    public.LAST(
        CASE WHEN itemid=227344 AND value='Yes' THEN 1 ELSE 0 END
    ) AS iv_saline_lock,
    public.LAST(
        CASE WHEN itemid=227345 THEN value ELSE NULL END
    ) AS gait_transferring,
    public.LAST(
        CASE WHEN itemid=227343 THEN value ELSE NULL END
    ) AS ambulatory_aid,
    public.LAST(
        CASE WHEN itemid=227346 THEN value ELSE NULL END
    ) AS mental_status,
    public.LAST(
        CASE WHEN itemid=227342 AND value='Yes' THEN 1 ELSE 0 END
    ) AS secondary_diagnosis,
    public.LAST(
        CASE WHEN itemid=227341 AND value='Yes' THEN 1 ELSE 0 END
    ) AS history_of_falling_3m,
    AVG(
        CASE WHEN itemid=227442 AND valuenum>0 AND valuenum<15 THEN valuenum ELSE NULL END
    ) AS potassium,
    AVG(
        CASE WHEN itemid=220645 AND valuenum>0 AND valuenum<250 THEN valuenum ELSE NULL END
    ) AS sodium,
    AVG(
        CASE WHEN itemid=220602 AND valuenum>0 AND valuenum<200 THEN valuenum ELSE NULL END
    ) AS chloride,
    AVG(
        CASE WHEN itemid=220615 AND valuenum>0 AND valuenum<66 THEN valuenum ELSE NULL END
    ) AS creatinine,
    AVG(
        CASE WHEN itemid=225624 AND valuenum>0 AND valuenum<275 THEN valuenum ELSE NULL END
    ) AS bun,
    AVG(
        CASE WHEN itemid=227443 AND valuenum>0 AND valuenum<66 THEN valuenum ELSE NULL END
    ) AS bicarbonate,
    AVG(
        CASE WHEN itemid=227073 AND valuenum>0 AND valuenum<55 THEN valuenum ELSE NULL END
    ) AS aniongap,
    AVG(
        CASE WHEN itemid=220545 AND valuenum<=100 THEN valuenum ELSE NULL END
    ) AS hematocrit,
    AVG(
        CASE WHEN itemid=220228 AND valuenum<30 THEN valuenum ELSE NULL END
    ) AS hemoglobin,
    AVG(
        CASE WHEN itemid=227457 AND valuenum>0 AND valuenum<2200 THEN valuenum ELSE NULL END
    ) AS platelet,
    AVG(
        CASE WHEN itemid=220546 AND valuenum>0 AND valuenum<1100 THEN valuenum ELSE NULL END
    ) AS wbc,
    AVG(
        CASE WHEN itemid=220635 AND valuenum>0 AND valuenum<22 THEN valuenum ELSE NULL END
    ) AS magnessium,
    public.LAST(
        CASE WHEN itemid=223752 AND valuenum>0 AND valuenum<=400 THEN valuenum ELSE NULL END
    ) AS blood_pressure_alarm_low,
    public.LAST(
        CASE WHEN itemid=223751 AND valuenum>0 AND valuenum<=400 THEN valuenum ELSE NULL END
    ) AS blood_pressure_alarm_high,
    AVG(
        CASE WHEN itemid=225677 AND valuenum>0 AND valuenum<22 THEN valuenum ELSE NULL END
    ) AS phosphate,
    AVG(
        CASE WHEN itemid=225625 AND valuenum>0 AND valuenum<22 THEN valuenum ELSE NULL END
    ) AS calcium,
    MAX(CASE WHEN itemid=223791 THEN value ELSE NULL END) AS pain_level,
    MAX(CASE WHEN itemid=224409 THEN value ELSE NULL END) AS pain_level_response,
    MAX(CASE WHEN itemid=228096 THEN value ELSE NULL END) AS richmond_ras_scale,
    MAX(CASE WHEN itemid=228299 THEN value ELSE NULL END) AS richmond_ras_scale_goal,
    AVG(
        CASE WHEN itemid=227465 AND valuenum>0 AND valuenum<=150 THEN valuenum ELSE NULL END
    ) AS pt,
    AVG(
        CASE WHEN itemid=227466 AND valuenum>0 AND valuenum<=150 THEN valuenum ELSE NULL END
    ) AS ptt,
    AVG(
        CASE WHEN itemid=227467 AND valuenum>0 AND valuenum<=30 THEN valuenum ELSE NULL END
    ) AS inr,
    public.LAST(
        CASE WHEN itemid=228305 THEN valuenum ELSE NULL END
    ) AS st_segment_monitoring_on,
    public.LAST(
        CASE WHEN itemid=225103 THEN valuenum ELSE NULL END
    ) AS iv_access_prior_to_admission,
    public.LAST(
        CASE WHEN itemid=227368 THEN valuenum ELSE NULL END
    ) AS gauge_20_dressing_occlusive,
    public.LAST(
        CASE WHEN itemid=228412 THEN valuenum ELSE NULL END
    ) AS strength_r_arm,
    public.LAST(
        CASE WHEN itemid=228409 THEN valuenum ELSE NULL END
    ) AS strength_l_arm,
    public.LAST(
        CASE WHEN itemid=228411 THEN valuenum ELSE NULL END
    ) AS strength_r_leg,
    public.LAST(
        CASE WHEN itemid=228410 THEN valuenum ELSE NULL END
    ) AS strength_l_leg,
    public.LAST(
        CASE WHEN itemid=226138 THEN valuenum ELSE NULL END
    ) AS gauge_20_placed_in_outside_facility,
    public.LAST(
        CASE WHEN itemid=228236 THEN valuenum ELSE NULL END
    ) AS insulin_pump,
    public.LAST(
        CASE WHEN itemid=225092 THEN valuenum ELSE NULL END
    ) AS self_activities_of_daily_living,
    public.LAST(
        CASE WHEN itemid=228100 THEN valuenum ELSE NULL END
    ) AS gauge_20_placed_in_the_field,
    public.LAST(
        CASE WHEN itemid=225094 THEN valuenum ELSE NULL END
    ) AS history_of_falls,
    public.LAST(
        CASE WHEN itemid=227349 THEN valuenum ELSE NULL END
    ) AS high_risk_gt_51_interventions,
    AVG(
        CASE WHEN itemid=225668 AND valuenum>0 AND valuenum<33 THEN valuenum ELSE NULL END
    ) AS lactate,
    public.LAST(
        CASE WHEN itemid=228648 THEN valuenum ELSE NULL END
    ) AS home_tube_feeding,
    public.LAST(
        CASE WHEN itemid=225106 THEN valuenum ELSE NULL END
    ) AS ethyl_alcohol,
    public.LAST(
        CASE WHEN itemid=228649 THEN valuenum ELSE NULL END
    ) AS pressure_ulcer_present,
    public.LAST(
        CASE WHEN itemid=225118 THEN valuenum ELSE NULL END
    ) AS difficulty_swallowing,
    public.LAST(
        CASE WHEN itemid=227367 THEN valuenum ELSE NULL END
    ) AS gauge_18_dressing_occlusive,
    public.LAST(
        CASE WHEN itemid=226137 THEN valuenum ELSE NULL END
    ) AS gauge_18_placed_in_outside_facility,
    public.LAST(
        CASE WHEN itemid=225184 THEN valuenum ELSE NULL END
    ) AS eye_care,
    public.LAST(
        CASE WHEN itemid=225087 THEN valuenum ELSE NULL END
    ) AS visual_hearing_deficit,
    public.LAST(
        CASE WHEN itemid=225113 THEN valuenum ELSE NULL END
    ) AS currently_experiencing_pain,
    public.LAST(
        CASE WHEN itemid=225126 THEN valuenum ELSE NULL END
    ) AS dialysis_patient
FROM mimiciv_icu.chartevents che
INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = che.stay_id
WHERE che.itemid IN (
    223951,
    224308,
    220274,
    220734,
    223830,
    228243,
    224641,
    224168,
    220047,
    220046,
    223770,
    223769,
    224161,
    224162,
    224054,
    224057,
    224055,
    224056,
    224058,
    224059,
    226253,
    227344,
    227345,
    227343,
    227346,
    227342,
    227341,
    227442,
    220645,
    220602,
    220615,
    225624,
    227443,
    227073,
    220545,
    220228,
    227457,
    220546,
    220635,
    223752,
    223751,
    225677,
    223791,
    224409,
    228096,
    228299,
    227465,
    227467,
    227466,
    228305,
    225103,
    227368,
    228412,
    228409,
    228411,
    228410,
    226138,
    228236,
    225092,
    228100,
    225094,
    227349,
    225668,
    228648,
    225106,
    228649,
    225118,
    227367,
    226137,
    225184,
    225087,
    225113,
    225126
)
GROUP BY che.subject_id, che.stay_id, che.charttime
ORDER BY che.charttime
