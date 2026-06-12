WITH
time_steps AS (
    SELECT
        icu.stay_id,
        time_at,
		DENSE_RANK() OVER (PARTITION BY icu.stay_id ORDER BY time_at ASC) AS step
    FROM
        mimiciv_icu.icustays icu,
        cohort_tmp co_tmp,
        generate_series(
            icu.intime,
            icu.outtime,
            '{time_step} {time_unit}'
        ) time_at
    WHERE co_tmp.stay_id=icu.stay_id
),
charts_bin AS (
    SELECT
        che.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            che.charttime,
            icu.intime
        ) time_at,
        COUNT(che.capillary_refill_rate_l) AS capillary_refill_rate_l,
        COUNT(che.capillary_refill_rate_r) AS capillary_refill_rate_r,
        COUNT(che.ph_blood) AS ph_blood,
        COUNT(che.ph_urine) AS ph_urine,
        COUNT(che.alarms_on) AS alarms_on,
        COUNT(che.parameters_checked) AS parameters_checked,
        COUNT(che.heart_rate_alarm_low) AS heart_rate_alarm_low,
        COUNT(che.heart_rate_alarm_high) AS heart_rate_alarm_high,
        COUNT(che.o2_saturation_alarm_low) AS o2_saturation_alarm_low,
        COUNT(che.o2_saturation_alarm_high) AS o2_saturation_alarm_high,
        COUNT(che.respiratory_rate_alarm_low) AS respiratory_rate_alarm_low,
        COUNT(che.respiratory_rate_alarm_high) AS respiratory_rate_alarm_high,
        COUNT(che.braden_sensory_perception) AS braden_sensory_perception,
        COUNT(che.braden_mobility) AS braden_mobility,
        COUNT(che.braden_moisture) AS braden_moisture,
        COUNT(che.braden_activity) AS braden_activity,
        COUNT(che.braden_nutrition) AS braden_nutrition,
        COUNT(che.braden_friction) AS braden_friction,
        COUNT(che.o2_saturation_desat_limit) AS o2_saturation_desat_limit,
        COUNT(che.iv_saline_lock) AS iv_saline_lock,
        COUNT(che.gait_transferring) AS gait_transferring,
        COUNT(che.ambulatory_aid) AS ambulatory_aid,
        COUNT(che.mental_status) AS mental_status,
        COUNT(che.secondary_diagnosis) AS secondary_diagnosis,
        COUNT(che.history_of_falling_3m) AS history_of_falling_3m,
        COUNT(che.potassium) AS potassium,
        COUNT(che.sodium) AS sodium,
        COUNT(che.chloride) AS chloride,
        COUNT(che.creatinine) AS creatinine,
        COUNT(che.bun) AS bun,
        COUNT(che.bicarbonate) AS bicarbonate,
        COUNT(che.aniongap) AS aniongap,
        COUNT(che.hematocrit) AS hematocrit,
        COUNT(che.hemoglobin) AS hemoglobin,
        COUNT(che.platelet) AS platelet,
        COUNT(che.wbc) AS wbc,
        COUNT(che.magnessium) AS magnesium,
        COUNT(che.blood_pressure_alarm_low) AS blood_pressure_alarm_low,
        COUNT(che.blood_pressure_alarm_high) AS blood_pressure_alarm_high,
        COUNT(che.phosphate) AS phosphate,
        COUNT(che.calcium) AS calcium,
        COUNT(che.pain_level) AS pain_level,
        COUNT(che.pain_level_response) AS pain_level_response,
        COUNT(che.richmond_ras_scale) AS richmond_ras_scale,
        COUNT(che.richmond_ras_scale_goal) AS richmond_ras_scale_goal,
        COUNT(che.pt) AS pt,
        COUNT(che.ptt) AS ptt,
        COUNT(che.inr) AS inr,
        COUNT(che.st_segment_monitoring_on) AS st_segment_monitoring_on,
        COUNT(che.gauge_20_dressing_occlusive) AS gauge_20_dressing_occlusive,
        COUNT(che.strength_r_arm) AS strength_r_arm,
        COUNT(che.strength_l_arm) AS strength_l_arm,
        COUNT(che.strength_r_leg) AS strength_r_leg,
        COUNT(che.strength_l_leg) AS strength_l_leg,
        COUNT(che.gauge_20_placed_in_outside_facility) AS gauge_20_placed_in_outside_facility,
        COUNT(che.gauge_20_placed_in_the_field) AS gauge_20_placed_in_the_field,
        COUNT(che.high_risk_gt_51_interventions) AS high_risk_gt_51_interventions,
        COUNT(che.lactate) AS lactate,
        COUNT(che.gauge_18_dressing_occlusive) AS gauge_18_dressing_occlusive,
        COUNT(che.gauge_18_placed_in_outside_facility) AS gauge_18_placed_in_outside_facility,
        COUNT(che.eye_care) AS eye_care
    FROM mimiciv_derived.charts che
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = che.stay_id
    GROUP BY che.stay_id, time_at
),
labs_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            lab.charttime,
            icu.intime
        ) time_at,
        COUNT(
            CASE WHEN itemid=50960 AND valuenum>0 AND valuenum<22 THEN valuenum ELSE NULL END
        ) AS magnessium,
        COUNT(
            CASE WHEN itemid=50970 AND valuenum>0 AND valuenum<22 THEN valuenum ELSE NULL END
        ) AS phosphate
    FROM mimiciv_hosp.labevents lab
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = lab.hadm_id
    WHERE lab.itemid IN (
        50960,
        50970
    )
    GROUP BY icu.stay_id, time_at
),
vents_bin AS (
    SELECT
        ven.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            ven.charttime,
            icu.intime
        ) time_at,
        COUNT(ven.fio2) fraction_inspired_o2,
        COUNT(CASE WHEN ven.respiratory_rate_set>=0 and ven.respiratory_rate_set<=50 THEN ven.respiratory_rate_set ELSE NULL END) respiratory_rate_set,
        COUNT(CASE WHEN ven.respiratory_rate_spontaneous<=70 THEN ven.respiratory_rate_spontaneous ELSE NULL END) respiratory_rate_spontaneous,
        COUNT(CASE WHEN ven.minute_volume<50 THEN ven.minute_volume ELSE NULL END) minute_volume,
        COUNT(CASE WHEN ven.tidal_volume_set<1500 THEN ven.tidal_volume_set ELSE NULL END) tidal_volume_set,
        COUNT(CASE WHEN ven.tidal_volume_observed<1500 THEN ven.tidal_volume_observed ELSE NULL END) tidal_volume_observed,
        COUNT(CASE WHEN ven.tidal_volume_spontaneous<1500 THEN ven.tidal_volume_spontaneous ELSE NULL END) tidal_volume_spontaneous,
        COUNT(CASE WHEN ven.plateau_pressure<70 THEN ven.plateau_pressure ELSE NULL END) plateau_pressure,
        COUNT(ven.peep) peep,
        COUNT(CASE WHEN ven.flow_rate>=0 THEN ven.flow_rate ELSE NULL END) flow_rate,
        COUNT(ven.ventilator_mode) ventilator_mode,
        COUNT(ven.ventilator_mode_hamilton) ventilator_mode_hamilton,
        COUNT(ven.ventilator_type) ventilator_type
    FROM mimiciv_derived.ventilator_setting ven
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = ven.stay_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY ven.stay_id, time_at
),
vitals_bin AS (
    SELECT
        vs.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            vs.charttime,
            icu.intime
        ) time_at,
        COUNT(vs.heart_rate) heart_rate,
        COUNT(vs.sbp) sbp,
        COUNT(vs.dbp) dbp,
        COUNT(vs.mbp) mbp,
        COUNT(vs.sbp_ni) sbp_ni,
        COUNT(vs.dbp_ni) dbp_ni,
        COUNT(vs.mbp_ni) mbp_ni,
        COUNT(vs.resp_rate) resp_rate,
        COUNT(vs.temperature) temperature,
        COUNT(vs.temperature_site) temperature_site,
        COUNT(vs.skin_temperature) skin_temperature,
        COUNT(vs.spo2) spo2,
        COUNT(vs.sao2) sao2,
        COUNT(CASE WHEN vs.glucose<2200 THEN vs.glucose ELSE NULL END) glucose
    FROM mimiciv_derived.vitalsign vs
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = vs.stay_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY vs.stay_id, time_at
),
chemistry_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            chm.charttime,
            icu.intime
        ) time_at,
        COUNT(chm.albumin) AS albumin,
        COUNT(chm.globulin) AS globulin,
        COUNT(chm.total_protein) AS total_protein,
        COUNT(CASE WHEN chm.aniongap>0 AND chm.aniongap<55 THEN chm.aniongap ELSE NULL END) AS aniongap,
        COUNT(CASE WHEN chm.bicarbonate>0 AND chm.bicarbonate<66 THEN chm.bicarbonate ELSE NULL END) AS bicarbonate,
        COUNT(CASE WHEN chm.bun>0 AND chm.bun<=275 THEN chm.bun ELSE NULL END) AS bun,
        COUNT(CASE WHEN chm.calcium>0 AND chm.calcium<22 THEN chm.calcium ELSE NULL END) AS calcium,
        COUNT(CASE WHEN chm.chloride>0 AND chm.chloride<200 THEN chm.chloride ELSE NULL END) AS chloride,
        COUNT(CASE WHEN chm.creatinine>0 AND chm.creatinine<66 THEN chm.creatinine ELSE NULL END) AS creatinine,
        COUNT(CASE WHEN chm.glucose<2200 THEN chm.glucose ELSE NULL END) glucose,
        COUNT(CASE WHEN chm.sodium>0 AND chm.sodium<250 THEN chm.sodium ELSE NULL END) AS sodium,
        COUNT(CASE WHEN chm.potassium>0 AND chm.potassium<15 THEN chm.potassium ELSE NULL END) AS potassium
    FROM mimiciv_derived.chemistry chm
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = chm.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icu.stay_id, time_at
),
bg_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            bg.charttime,
            icu.intime
        ) time_at,
        -- oxygen related parameters
        COUNT(bg.so2) AS so2,
        COUNT(CASE WHEN bg.po2>0 AND bg.po2<770 THEN bg.po2 ELSE NULL END) AS po2,
        COUNT(CASE WHEN bg.pco2>0 AND bg.pco2<220 THEN bg.pco2 ELSE NULL END) AS pco2,
        COUNT(bg.fio2_chartevents) AS fio2_ch,
        COUNT(bg.fio2) AS fio2,
        COUNT(bg.aado2) AS aado2,
        COUNT(bg.aado2_calc) AS aado2_calc,
        COUNT(bg.pao2fio2ratio) AS pao2fio2ratio,
        COUNT(CASE WHEN bg.ph>6 AND bg.ph<10 THEN bg.ph ELSE NULL END) AS ph,
        COUNT(CASE WHEN bg.baseexcess>=-50 AND bg.baseexcess<=50 THEN bg.baseexcess ELSE NULL END) AS base_excess,
        COUNT(CASE WHEN bg.bicarbonate>0 AND bg.bicarbonate<66 THEN bg.bicarbonate ELSE NULL END) AS bicarbonate,
        COUNT(CASE WHEN bg.totalco2>0 AND bg.totalco2<=300 THEN bg.totalco2 ELSE NULL END) AS totalco2,
        -- blood count parameters
        COUNT(bg.hematocrit) AS hematocrit,
        COUNT(CASE WHEN bg.hemoglobin<30 THEN bg.hemoglobin ELSE NULL END) AS hemoglobin,
        COUNT(bg.carboxyhemoglobin) AS carboxyhemoglobin,
        COUNT(bg.methemoglobin) AS methemoglobin,
        -- chemistry
        COUNT(CASE WHEN bg.chloride>0 AND bg.chloride<200 THEN bg.chloride ELSE NULL END) AS chloride,
        COUNT(CASE WHEN bg.calcium>0 AND bg.calcium<5 THEN bg.calcium ELSE NULL END) AS calcium,
        COUNT(CASE WHEN bg.temperature>0 AND bg.temperature<=60 THEN bg.temperature ELSE NULL END) AS temperature,
        COUNT(CASE WHEN bg.potassium>0 AND bg.potassium<15 THEN bg.potassium ELSE NULL END) AS potassium,
        COUNT(CASE WHEN bg.sodium>0 AND bg.sodium<250 THEN bg.sodium ELSE NULL END) AS sodium,
        COUNT(CASE WHEN bg.lactate>0 AND bg.lactate<33 THEN bg.lactate ELSE NULL END) AS lactate,
        COUNT(CASE WHEN bg.glucose<2200 THEN bg.glucose ELSE NULL END) glucose
    FROM mimiciv_derived.bg
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = bg.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icu.stay_id, time_at
),
bc_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            bc.charttime,
            icu.intime
        ) time_at,
        COUNT(bc.hematocrit) AS hematocrit,
        COUNT(CASE WHEN bc.hemoglobin<30 THEN bc.hemoglobin ELSE NULL END) AS hemoglobin,
        COUNT(CASE WHEN bc.mch>=0 AND bc.mch<=50 THEN bc.mch ELSE NULL END) AS mch,
        COUNT(CASE WHEN bc.mchc>0 AND bc.mch<=70 THEN bc.mchc ELSE NULL END) AS mchc,
        COUNT(bc.mcv) AS mcv,
        COUNT(CASE WHEN bc.platelet>0 AND bc.platelet<2200 THEN bc.platelet ELSE NULL END) AS platelet,
        COUNT(CASE WHEN bc.rbc>=0 AND bc.rbc<=10 THEN bc.rbc ELSE NULL END) AS rbc,
        COUNT(CASE WHEN bc.rdw>0 AND bc.rdw<=50 THEN bc.rdw ELSE NULL END) AS rdw,
        COUNT(bc.rdwsd) AS rdwsd,
        COUNT(CASE WHEN bc.wbc>0 AND bc.wbc<1100 THEN bc.wbc ELSE NULL END) AS wbc
    FROM mimiciv_derived.complete_blood_count bc
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = bc.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icu.stay_id, time_at
),
bd_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            bd.charttime,
            icu.intime
        ) time_at,
        COUNT(CASE WHEN bd.wbc>0 AND bd.wbc<1100 THEN bd.wbc ELSE NULL END) AS wbc,
        COUNT(bd.basophils_abs) AS basophils_abs,
        COUNT(bd.eosinophils_abs) AS eosinophils_abs,
        COUNT(bd.lymphocytes_abs) AS lymphocytes_abs,
        COUNT(bd.monocytes_abs) AS monocytes_abs,
        COUNT(bd.neutrophils_abs) AS neutrophils_abs,
        COUNT(bd.basophils) AS basophils,
        COUNT(bd.eosinophils) AS eosinophils,
        COUNT(bd.lymphocytes) AS lymphocytes,
        COUNT(bd.monocytes) AS monocytes,
        COUNT(bd.neutrophils) AS neutrophils,
        COUNT(bd.atypical_lymphocytes) AS atypical_lymphocytes,
        COUNT(bd.bands) AS bands,
        COUNT(bd.immature_granulocytes) AS immature_granulocytes,
        COUNT(bd.metamyelocytes) AS metamyelocytes,
        COUNT(CASE WHEN bd.nrbc>=0 AND bd.nrbc<=100 THEN bd.nrbc ELSE NULL END) AS nrbc
    FROM mimiciv_derived.blood_differential bd
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = bd.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icu.stay_id, time_at
),
coagulation_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            coa.charttime,
            icu.intime
        ) time_at,
        COUNT(CASE WHEN coa.pt>0 AND coa.pt<=150 THEN coa.pt ELSE NULL END) AS pt,
        COUNT(CASE WHEN coa.ptt>0 AND coa.ptt<=150 THEN coa.ptt ELSE NULL END) AS ptt,
        COUNT(coa.d_dimer) AS d_dimer,
        COUNT(coa.fibrinogen) AS fibrinogen,
        COUNT(CASE WHEN coa.inr>0 AND coa.inr<=150 THEN coa.inr ELSE NULL END) AS inr,
        COUNT(coa.thrombin) AS thrombin
    FROM mimiciv_derived.coagulation coa
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = coa.hadm_id
    GROUP BY icu.stay_id, time_at
),
o2_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            o2.charttime,
            icu.intime
        ) time_at,
        COUNT(CASE WHEN o2.o2_flow>=0 AND o2.o2_flow<=100 THEN o2.o2_flow ELSE NULL END) AS o2_flow,
        COUNT(CASE WHEN o2.o2_flow_additional>=0 AND o2.o2_flow_additional<=100 THEN o2.o2_flow_additional ELSE NULL END) AS o2_flow_additional,
        COUNT(o2.o2_delivery_device_1) AS o2_delivery_device_1,
        COUNT(o2.o2_delivery_device_2) AS o2_delivery_device_2,
        COUNT(o2.o2_delivery_device_3) AS o2_delivery_device_3,
        COUNT(o2.o2_delivery_device_4) AS o2_delivery_device_4
    FROM mimiciv_derived.oxygen_delivery o2
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = o2.stay_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icu.stay_id, time_at
),
gcs_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            gcs.charttime,
            icu.intime
        ) time_at,
        COUNT(gcs.gcs) AS gcs,
        COUNT(gcs.gcs_motor) AS gcs_motor,
        COUNT(gcs.gcs_verbal) AS gcs_verbal,
        COUNT(gcs.gcs_eyes) AS gcs_eyes,
        COUNT(gcs.gcs_unable) AS gcs_unable
    FROM mimiciv_derived.gcs
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = gcs.stay_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icu.stay_id, time_at
),
weight_stat AS (
    SELECT
        wd.stay_id,
        (PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER BY weight)) AS q1,
        (PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER BY weight)) AS q3,
        (PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY weight)) AS median,
        ((PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER BY weight)) -
            (PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER BY weight))) AS iqr
    FROM mimiciv_derived.weight_durations wd
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=wd.stay_id
    GROUP BY wd.stay_id
),
weight_bin AS (
    SELECT
        wtd.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            wtd.starttime,
            icu.intime
        ) time_at,
        COUNT(wtd.weight) weight
    FROM mimiciv_derived.weight_durations wtd
    JOIN weight_stat w_stat ON w_stat.stay_id = wtd.stay_id
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = wtd.stay_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    WHERE
    wtd.weight BETWEEN (w_stat.q1 - 1.5 * w_stat.iqr) AND (w_stat.q3 + 1.5 * w_stat.iqr) AND
    ABS((wtd.weight - w_stat.median) / w_stat.median) < 0.2
    GROUP BY wtd.stay_id, time_at
),
cm_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            cm.charttime,
            icu.intime
        ) time_at,
        COUNT(CASE WHEN cm.troponin_t>0 AND cm.troponin_t<24 THEN cm.troponin_t ELSE NULL END) AS troponin_t,
        COUNT(cm.ck_mb) AS ck_mb,
        COUNT(cm.ntprobnp) AS ntprobnp
    FROM mimiciv_derived.cardiac_marker cm
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = cm.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icu.stay_id, time_at
),
enz_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            enz.charttime,
            icu.intime
        ) time_at,
        COUNT(CASE WHEN enz.alt>0 AND enz.alt<11000 THEN enz.alt ELSE NULL END) AS alt,
        COUNT(enz.alp) AS alp,
        COUNT(CASE WHEN enz.ast>0 AND enz.ast<22000 THEN enz.ast ELSE NULL END) AS ast,
        COUNT(enz.amylase) As amylase,
        COUNT(CASE WHEN enz.bilirubin_total>0 AND enz.bilirubin_total<66 THEN enz.bilirubin_total ELSE NULL END) AS bilirubin_total,
        COUNT(CASE WHEN enz.bilirubin_direct>0 AND enz.bilirubin_direct<66 THEN enz.bilirubin_direct ELSE NULL END) AS bilirubin_direct,
        COUNT(CASE WHEN enz.bilirubin_indirect>0 AND enz.bilirubin_indirect<66 THEN enz.bilirubin_indirect ELSE NULL END) AS bilirubin_indirect,
        COUNT(enz.ck_cpk) AS ck_cpk,
        COUNT(enz.ck_mb) AS ck_mb,
        COUNT(enz.ggt) AS ggt,
        COUNT(CASE WHEN enz.ld_ldh>0 AND enz.ld_ldh<35000 THEN enz.ld_ldh ELSE NULL END) AS ld_ldh
    FROM mimiciv_derived.enzyme enz
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = enz.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icu.stay_id, time_at
),
icp_bin AS (
    SELECT
        icp.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            icp.charttime,
            icu.intime
        ) time_at,
        COUNT(icp.icp) icp
    FROM mimiciv_derived.icp icp
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = icp.stay_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icp.stay_id, time_at
),
inf_bin AS (
    SELECT
        icu.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            inf.charttime,
            icu.intime
        ) time_at,
        COUNT(inf.crp) crp
    FROM mimiciv_derived.inflammation inf
    INNER JOIN mimiciv_icu.icustays icu ON icu.hadm_id = inf.hadm_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY icu.stay_id, time_at
),
rhythm_bin AS (
    SELECT
        rhythm.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            rhythm.charttime,
            icu.intime
        ) time_at,
        COUNT(rhythm.heart_rhythm) AS heart_rhythm,
        COUNT(rhythm.ectopy_type) AS ectopy_type,
        COUNT(rhythm.ectopy_frequency) AS ectopy_frequency,
        COUNT(rhythm.ectopy_type_secondary) AS ectopy_type_secondary,
        COUNT(rhythm.ectopy_frequency_secondary) AS ectopy_frequency_secondary
    FROM mimiciv_derived.rhythm rhythm
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = rhythm.stay_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY rhythm.stay_id, time_at
),
uo_bin AS (
    SELECT
        uo.stay_id,
        date_bin(
            INTERVAL '{time_step} {time_unit}',
            uo.charttime,
            icu.intime
        ) time_at,
        COUNT(CASE WHEN uo.urineoutput>=0 AND uo.urineoutput<2445 THEN uo.urineoutput ELSE 0 END) AS urine_output
    FROM mimiciv_derived.urine_output uo
    INNER JOIN mimiciv_icu.icustays icu ON icu.stay_id = uo.stay_id
    INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id=icu.stay_id
    GROUP BY uo.stay_id, time_at
)
SELECT
    ts.stay_id,
	ts.step,
	-- Harutyunyan et al. (2019)
	che.capillary_refill_rate_l,
	che.capillary_refill_rate_r,
    vs.dbp AS diastolic_blood_pressure,
    (SELECT SUM(v) FROM (VALUES (ven.fraction_inspired_o2),(bg.fio2)) T(v)) AS fraction_inspired_o2,
    gcs.gcs_eyes,
    gcs.gcs_verbal,
    gcs.gcs_motor,
    gcs.gcs,
    gcs.gcs_unable,
    (SELECT SUM(v) FROM (VALUES (vs.glucose),(chm.glucose),(bg.glucose)) T(v)) AS glucose,
    vs.heart_rate,
    vs.mbp AS mean_blood_pressure,
    (SELECT SUM(v) FROM (VALUES (bg.so2),(vs.sao2)) T(v)) AS arterial_o2_saturation,
    vs.spo2 AS peripheral_o2_saturation,
    vs.resp_rate AS respiratory_rate,
    vs.sbp AS systolic_blood_pressure,
    vs.temperature,
    vs.skin_temperature,
    wtd.weight,
    (SELECT SUM(v) FROM (VALUES (che.ph_blood),(bg.ph)) T(v)) AS ph_blood,
    che.ph_urine,
    --
    che.alarms_on,
    che.parameters_checked,
    che.heart_rate_alarm_low,
    che.heart_rate_alarm_high,
    che.o2_saturation_alarm_low,
    che.o2_saturation_alarm_high,
    che.respiratory_rate_alarm_low,
    che.respiratory_rate_alarm_high,
    che.braden_sensory_perception,
    che.braden_mobility,
    che.braden_moisture,
    che.braden_activity,
    che.braden_nutrition,
    che.braden_friction,
    che.o2_saturation_desat_limit,
    che.iv_saline_lock,
    che.gait_transferring,
    che.ambulatory_aid,
    che.mental_status,
    che.secondary_diagnosis,
    che.history_of_falling_3m,
    (SELECT SUM(v) FROM (VALUES (chm.potassium),(bg.potassium),(che.potassium)) T(v)) AS potassium,
    (SELECT SUM(v) FROM (VALUES (chm.sodium),(bg.sodium),(che.sodium)) T(v)) AS sodium,
    (SELECT SUM(v) FROM (VALUES (chm.chloride),(bg.chloride),(che.chloride)) T(v)) AS chloride,
    (SELECT SUM(v) FROM (VALUES (chm.creatinine),(che.creatinine)) T(v)) AS creatinine,
    (SELECT SUM(v) FROM (VALUES (chm.bun),(che.bun)) T(v)) AS blood_urea_nitrogen,
    (SELECT SUM(v) FROM (VALUES (chm.bicarbonate),(bg.bicarbonate),(che.bicarbonate)) T(v)) AS bicarbonate,
    (SELECT SUM(v) FROM (VALUES (chm.aniongap),(che.aniongap)) T(v)) AS aniongap,
    (SELECT SUM(v) FROM (VALUES (bg.hematocrit),(bc.hematocrit),(che.hematocrit)) T(v)) AS hematocrit,
    (SELECT SUM(v) FROM (VALUES (bg.hemoglobin),(bc.hemoglobin),(che.hemoglobin)) T(v)) AS hemoglobin,
    (SELECT SUM(v) FROM (VALUES (bc.platelet),(che.platelet)) T(v)) AS platelet_count,
    (SELECT SUM(v) FROM (VALUES (bc.wbc),(che.wbc),(bd.wbc)) T(v)) AS white_blood_cells,
    (SELECT SUM(v) FROM (VALUES (che.magnesium),(lab.magnessium)) T(v)) AS magnesium,
    che.blood_pressure_alarm_low,
    che.blood_pressure_alarm_high,
    (SELECT SUM(v) FROM (VALUES (che.phosphate),(lab.phosphate)) T(v)) AS phosphate,
    (SELECT SUM(v) FROM (VALUES (chm.calcium),(che.calcium)) T(v)) AS calcium,
    bg.calcium AS free_calcium,
    che.pain_level,
    che.pain_level_response,
    che.richmond_ras_scale,
    che.richmond_ras_scale_goal,
    (SELECT SUM(v) FROM (VALUES (coa.pt),(che.pt)) T(v)) AS prothrombin_time,
    (SELECT SUM(v) FROM (VALUES (coa.inr),(che.inr)) T(v)) AS international_normalized_ratio,
    (SELECT SUM(v) FROM (VALUES (coa.ptt),(che.ptt)) T(v)) AS partial_thromboplastin_time,
    che.st_segment_monitoring_on,
    o2.o2_flow,
    o2.o2_flow_additional,
    o2.o2_delivery_device_1,
    o2.o2_delivery_device_2,
    o2.o2_delivery_device_3,
    o2.o2_delivery_device_4,
    che.gauge_20_dressing_occlusive,
    che.strength_r_arm,
    che.strength_l_arm,
    che.strength_r_leg,
    che.strength_l_leg,
    che.gauge_20_placed_in_outside_facility,
    che.gauge_20_placed_in_the_field,
    che.high_risk_gt_51_interventions,
    (SELECT SUM(v) FROM (VALUES (che.lactate),(bg.lactate)) T(v)) AS lactate,
    che.gauge_18_dressing_occlusive,
    che.gauge_18_placed_in_outside_facility,
    che.eye_care,
    bc.rbc AS red_blood_cells,
    bc.mcv AS mean_corpuscular_volume,
    bc.mch AS mean_corpuscular_hemoglobin,
    bc.mchc AS mean_corpuscular_hemoglobin_concentration,
    bc.rdw AS red_cell_distribution_width,
    bg.base_excess,
    bg.po2 AS partial_pressure_of_o2,
    bg.pco2 AS partial_pressure_of_co2,
    bg.totalco2 AS total_co2,
    bg.aado2 AS alveolar_arterial_gradient,
    bg.carboxyhemoglobin,
    bg.methemoglobin,
    bg.temperature AS blood_gas_temperature,
    chm.albumin,
    chm.globulin,
    chm.total_protein,
    vs.temperature_site,
    bd.basophils_abs,
    bd.eosinophils_abs,
    bd.lymphocytes_abs,
    bd.monocytes_abs,
    bd.neutrophils_abs,
    bd.basophils,
    bd.eosinophils,
    bd.lymphocytes,
    bd.monocytes,
    bd.neutrophils,
    bd.atypical_lymphocytes,
    bd.bands,
    bd.immature_granulocytes,
    bd.metamyelocytes,
    bd.nrbc AS nucleated_red_cells,
    cm.troponin_t,
    cm.ck_mb AS creatinine_kinase_mb_isoenzyme,
    cm.ntprobnp AS n_terminal_pro_hormone_bnp,
    coa.d_dimer,
    coa.fibrinogen,
    coa.thrombin,
    enz.alt AS alanine_aminotransferase,
    enz.alp AS alkaline_phosphatase,
    enz.ast AS asparate_aminotransferase,
    enz.amylase,
    enz.bilirubin_total,
    enz.bilirubin_indirect,
    enz.bilirubin_direct,
    enz.ck_cpk AS creatinine_kinase,
    enz.ggt AS gamma_glutamyltransferase,
    enz.ld_ldh AS lactate_dehydrogenase,
    icp.icp AS intra_cranial_pressure,
    inf.crp AS c_reactive_protein,
    rhythm.heart_rhythm,
    rhythm.ectopy_type,
    rhythm.ectopy_frequency,
    rhythm.ectopy_type_secondary,
    rhythm.ectopy_frequency_secondary,
    uo.urine_output,
    ven.respiratory_rate_set,
    ven.respiratory_rate_spontaneous,
    ven.minute_volume,
    ven.tidal_volume_set,
    ven.tidal_volume_observed,
    ven.tidal_volume_spontaneous,
    ven.plateau_pressure,
    ven.peep,
    ven.flow_rate,
    ven.ventilator_mode,
    ven.ventilator_mode_hamilton,
    ven.ventilator_type
FROM time_steps ts
INNER JOIN cohort_tmp co_tmp ON co_tmp.stay_id = ts.stay_id
LEFT JOIN charts_bin che ON ts.stay_id = che.stay_id AND ts.time_at = che.time_at
LEFT JOIN labs_bin lab ON ts.stay_id = lab.stay_id AND ts.time_at = lab.time_at
LEFT JOIN vents_bin ven ON ts.stay_id = ven.stay_id AND ts.time_at = ven.time_at
LEFT JOIN vitals_bin vs ON ts.stay_id = vs.stay_id AND ts.time_at = vs.time_at
LEFT JOIN chemistry_bin chm ON ts.stay_id = chm.stay_id AND ts.time_at = chm.time_at
LEFT JOIN bg_bin bg ON ts.stay_id = bg.stay_id AND ts.time_at = bg.time_at
LEFT JOIN bc_bin bc ON ts.stay_id = bc.stay_id AND ts.time_at = bc.time_at
LEFT JOIN bd_bin bd ON ts.stay_id = bd.stay_id AND ts.time_at = bd.time_at
LEFT JOIN coagulation_bin coa ON ts.stay_id = coa.stay_id AND ts.time_at = coa.time_at
LEFT JOIN o2_bin o2 ON ts.stay_id = o2.stay_id AND ts.time_at = o2.time_at
LEFT JOIN gcs_bin gcs ON ts.stay_id = gcs.stay_id AND ts.time_at = gcs.time_at
LEFT JOIN weight_bin wtd ON ts.stay_id = wtd.stay_id AND ts.time_at = wtd.time_at
LEFT JOIN cm_bin cm ON ts.stay_id = cm.stay_id AND ts.time_at = cm.time_at
LEFT JOIN enz_bin enz ON ts.stay_id = enz.stay_id AND ts.time_at = enz.time_at
LEFT JOIN icp_bin icp ON ts.stay_id = icp.stay_id AND ts.time_at = icp.time_at
LEFT JOIN inf_bin inf ON ts.stay_id = inf.stay_id AND ts.time_at = inf.time_at
LEFT JOIN rhythm_bin rhythm ON ts.stay_id = rhythm.stay_id AND ts.time_at = rhythm.time_at
LEFT JOIN uo_bin uo ON ts.stay_id = uo.stay_id AND ts.time_at = uo.time_at
ORDER BY ts.stay_id, ts.step
