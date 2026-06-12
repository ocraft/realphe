from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from xarray import DataArray, Dataset
import xarray as xr

from .query import MimicDb, MimicSql


@dataclass
class CohortExclusion:
    # pylint: disable=too-many-instance-attributes
    exclusion_short_icu_los: int = 0
    exclusion_long_icu_los: int = 0
    exclusion_not_adult: int = 0
    exclusion_not_first_icu_stay: int = 0
    exclusion_not_first_hosp_stay: int = 0
    exclusion_surgical: int = 0
    exclusion_many_services: int = 0
    exclusion_invalid_data: int = 0
    exclusion_only_rare_diagnoses: int = 0

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class CohortSetup:
    adult_age: int = 16
    short_icu_los_hours: int = 12
    long_icu_los_hours: int = 336
    diagnoses_min_samples: int = 300
    diagnoses_coding: str = 'icd9'
    exclusion: CohortExclusion = field(default_factory=CohortExclusion)
    features: Optional[List[str]] = None
    outcomes: Optional[List[str]] = None


def cohort(source: MimicDb | pd.DataFrame, cohort_setup: Optional[CohortSetup] = None) -> Dataset:
    if cohort_setup is None:
        cohort_setup = CohortSetup()
    criteria = asdict(cohort_setup)

    if isinstance(source, MimicDb):
        co = load_cohort(source, cohort_setup)
    else:
        co = source

    for exclusion in criteria['exclusion']:
        to_exclude = criteria['exclusion'][exclusion]
        if to_exclude:
            co = co.loc[~(co[exclusion] == to_exclude)]

    co = _encode_categories(co, co.select_dtypes(include=['category']).columns.tolist())
    co = _encode_time(co, co.select_dtypes(include=['datetime']).columns.tolist())

    if cohort_setup.features is not None and cohort_setup.outcomes is not None:
        co = co[cohort_setup.features + cohort_setup.outcomes]
    elif cohort_setup.features is not None:
        co = co[cohort_setup.features]
    elif cohort_setup.outcomes is not None:
        co = co[cohort_setup.outcomes]

    co.index.names = ['sample']

    co = co.to_xarray()
    if cohort_setup.features is not None:
        co.attrs['cohort_features'] = list(cohort_setup.features)
    if cohort_setup.outcomes is not None:
        co.attrs['outcomes'] = list(cohort_setup.outcomes)

    return co


def load_cohort(db: MimicDb, cohort_setup: Optional[CohortSetup] = None) -> pd.DataFrame:
    if cohort_setup is None:
        cohort_setup = CohortSetup()
    criteria = asdict(cohort_setup)

    co = db.execute(MimicSql.COHORT, criteria, dtype={
        'subject_id': 'int32',
        'hadm_id': 'int32',
        'stay_id': 'int32',
        'icustay_seq': 'int32',
        'intime_hr': 'datetime64[ns]',
        'outtime_hr': 'datetime64[ns]',
        # Outcomes
        'hospital_expire_flag': 'category',
        'icu_expire_flag': 'category',
        'thirtyday_expire_flag': 'category',
        'los_icu_hours': 'float32',
        'los_hospital_hours': 'float32',
        'deathtime_hours': 'float32',
        # Demography
        'age': 'float32',
        'language': 'category',
        'gender': 'category',
        'height': 'float32',
        'insurance': 'category',
        'marital_status': 'category',
        'race': 'category',
        # Admission
        'admission_location': 'category',
        'admission_time': 'datetime64[ns]',
        'admission_type': 'category',
        'first_careunit': 'category',
        'hospstay_seq': 'int32',
        'last_service': 'category',
        # admission history
        'currently_experiencing_pain': 'category',
        'dialysis_patient': 'category',
        'difficulty_swallowing': 'category',
        'ethyl_alcohol': 'category',
        'history_of_falls': 'category',
        'home_tube_feeding': 'category',
        'insulin_pump': 'category',
        'iv_access_prior_to_admission': 'category',
        'pressure_ulcer_present': 'category',
        'self_activities_of_daily_living': 'category',
        'visual_hearing_deficit': 'category',
        # exclusions
        'exclusion_short_icu_los': 'category',
        'exclusion_long_icu_los': 'category',
        'exclusion_not_adult': 'category',
        'exclusion_not_first_icu_stay': 'category',
        'exclusion_not_first_hosp_stay': 'category',
        'exclusion_surgical': 'category',
        'exclusion_many_services': 'category',
        'exclusion_invalid_data': 'category'
    })
    co.set_index('stay_id', inplace=True)

    co['exclusion_only_rare_diagnoses'] = 0
    cohort_set = cohort(co, cohort_setup)
    with set_cohort(db, cohort_set):
        co.loc[cohort_set.sample, 'exclusion_only_rare_diagnoses'] = 1
        co.loc[(
            cohort_set
            .merge(diagnoses(db, criteria), join="outer")
            .dropna(dim='sample', subset=['target'])
            .sample

        ), 'exclusion_only_rare_diagnoses'] = 0

    co['exclusion_only_rare_diagnoses'] = co['exclusion_only_rare_diagnoses'].astype('category')
    return co


def _encode_categories(df: pd.DataFrame, cat_fea: List[str]) -> pd.DataFrame:
    # NaN will be encoded to -1
    for fea in cat_fea:
        if df[fea] is not None:
            df[fea] = df[fea].cat.codes
    return df


def _encode_time(df: pd.DataFrame, time_fea: List[str]) -> pd.DataFrame:
    for fea in time_fea:
        if df[fea] is not None:
            df[fea] = df[fea].astype(int).astype('int32')
    return df


@contextmanager
def set_cohort(db: MimicDb, cohort_set: DataArray | Dataset | pd.DataFrame):
    # prepare in-memory csv buffer with cohort stay_id list
    db.con.commit()
    trans = db.con.begin()
    buffer = StringIO()
    cur = None
    try:
        if isinstance(cohort_set, pd.DataFrame):
            samples = cohort_set.index
        else:
            samples = cohort_set.get_index('sample')
        np.savetxt(buffer, samples, fmt='%s', delimiter=',')
        buffer.seek(0)

        # prepare temporary table with cohort stay_id
        cur = db.con.connection.cursor()
        cur.execute('CREATE TEMPORARY TABLE cohort_tmp(stay_id INT) ON COMMIT DROP')
        cur.copy_from(buffer, 'cohort_tmp', sep=',')  # pyright: ignore[reportGeneralTypeIssues]

        yield cur
    except:
        trans.rollback()
        raise
    finally:
        trans.commit()
        buffer.close()
        if cur is not None:
            cur.close()


def diagnoses(db: MimicDb, criteria: Optional[Dict[str, str]] = None) -> Dataset:
    coding = criteria['diagnoses_coding'] if criteria and criteria['diagnoses_coding'] else 'icd9'
    if coding == 'icd9':
        dx = icd9(db, criteria)
    elif coding == 'ccs':
        dx = ccs(db, criteria)
    elif coding == 'phecode':
        dx = phecode(db, criteria)
    else:
        raise ValueError(f'Unknown coding system: {coding}')

    dx.index.names = ['sample', 'diagnose']
    dx.columns.name = coding

    dx_set = dx.to_xarray()

    mlb = MultiLabelBinarizer()
    dx_set = dx_set.assign(target=(
        ['sample', 'label'],
        mlb.fit_transform(tuple(a[~pd.isnull(a)] for a in dx_set[coding].values)).astype(np.int8)
    ))

    return dx_set.assign_coords({'label': mlb.classes_})


def icd9(db: MimicDb, criteria: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    criteria_base = {
        'diagnoses_min_samples': '0'
    }
    criteria_final = prepare_criteria(criteria_base, criteria)

    dx = db.execute(MimicSql.DIAGNOSES_ICD9, criteria_final, dtype={
        'stay_id': 'int32',
        'dx_seq_num': 'int8',
        'icd9': 'category',
        'icd9_name': 'category',
    })
    dx.set_index(['stay_id', 'dx_seq_num'], inplace=True)

    return dx


def ccs(db: MimicDb, criteria: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    criteria_base = {
        'diagnoses_min_samples': '0'
    }
    criteria_final = prepare_criteria(criteria_base, criteria)

    dx = db.execute(MimicSql.DIAGNOSES_CCS, criteria_final, dtype={
        'stay_id': 'int32',
        'dx_seq_num': 'int8',
        'ccs': 'category',
        'ccs_name': 'category'
    })
    dx.set_index(['stay_id', 'dx_seq_num'], inplace=True)

    return dx


def phecode(db: MimicDb, criteria: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    criteria_base = {
        'diagnoses_min_samples': '0'
    }
    criteria_final = prepare_criteria(criteria_base, criteria)

    dx = db.execute(MimicSql.DIAGNOSES_PHECODE, criteria_final, dtype={
        'stay_id': 'int32',
        'dx_seq_num': 'int8',
        'phecode': 'category'
    })
    dx.set_index(['stay_id', 'dx_seq_num'], inplace=True)

    return dx


def signals(
    source: MimicDb | pd.DataFrame,
    criteria: Optional[Dict[str, str]] = None
) -> Dataset:
    if isinstance(source, MimicDb):
        ts = time_series_state(source, criteria)
    else:
        ts = source

    ts = ts.reset_index()

    # categories with meaningful order
    ts['ectopy_frequency'] = pd.Categorical(
        ts['ectopy_frequency'],
        categories=['None', 'Rare', 'Occasional', 'Frequent', 'Runs Vtach'],
        ordered=True)
    ts['ectopy_frequency_secondary'] = pd.Categorical(
        ts['ectopy_frequency_secondary'],
        categories=['None', 'Rare', 'Occasional', 'Frequent', 'Runs Vtach'],
        ordered=True)
    ts['braden_activity'] = pd.Categorical(
        ts['braden_activity'],
        categories=['Walks Frequently', 'Walks Occasionally', 'Chairfast', 'Bedfast'],
        ordered=True)
    ts['braden_friction'] = pd.Categorical(
        ts['braden_friction'],
        categories=['No Apparent Problem', 'Potential Problem', 'Problem'],
        ordered=True)
    ts['braden_mobility'] = pd.Categorical(
        ts['braden_mobility'],
        categories=['No Limitations', 'Slight Limitations', 'Very Limited', 'Completely Immobile'],
        ordered=True)
    ts['braden_moisture'] = pd.Categorical(
        ts['braden_moisture'],
        categories=['Rarely Moist', 'Occasionally Moist', 'Consistently Moist', 'Moist'],
        ordered=True)
    ts['braden_nutrition'] = pd.Categorical(
        ts['braden_nutrition'],
        categories=['Excellent', 'Adequate', 'Probably Inadequate', 'Very Poor'],
        ordered=True)
    ts['braden_sensory_perception'] = pd.Categorical(
        ts['braden_sensory_perception'],
        categories=['No Impairment', 'Slight Impairment', 'Very Limited', 'Completely Limited'],
        ordered=True)
    ts['skin_temperature'] = pd.Categorical(
        ts['skin_temperature'],
        categories=['Cold', 'Cool', 'Warm', 'Hot'],
        ordered=True)
    ts['ambulatory_aid'] = pd.Categorical(
        ts['ambulatory_aid'],
        categories=['None', 'Furniture', 'Cane', 'Crutches',
                    'Walker', 'Nurse assist', 'Wheel chair', 'Bed rest'],
        ordered=True)
    ts['gait_transferring'] = pd.Categorical(
        ts['gait_transferring'],
        categories=['Normal ', 'Weak', 'Impaired', 'Immobile', 'Bed rest'],
        ordered=True)
    ts['pain_level'] = pd.Categorical(
        ts['pain_level'],
        categories=['None', 'None to Mild', 'Mild ', 'Mild to Moderate', 'Moderate',
                    'Moderate to Severe', 'Severe', 'Severe to Worse', 'Worst', 'Unable to Score'],
        ordered=True)
    ts['pain_level_response'] = pd.Categorical(
        ts['pain_level_response'],
        categories=['None', 'None to Mild', 'Mild ', 'Mild to Moderate', 'Moderate',
                    'Moderate to Severe', 'Severe', 'Severe to Worse', 'Worst', 'Unable to Score'],
        ordered=True)
    ts['richmond_ras_scale'] = pd.Categorical(
        ts['richmond_ras_scale'],
        categories=[
            '-5 Unarousable, no response to voice or physical stimulation'
            '-4 Deep sedation, no response to voice, but movement or eye opening to physical stimulation',
            '-3 Moderate sedation, movement or eye opening; No eye contact',
            '-2 Light sedation, briefly awakens to voice (eye opening/contact) < 10 sec',
            '-1 Awakens to voice (eye opening/contact) > 10 sec',
            ' 0  Alert and calm',
            '+1 Anxious, apprehensive, but not aggressive',
            '+2 Frequent nonpurposeful movement, fights ventilator',
            '+3 Pulls or removes tube(s) or catheter(s); aggressive',
            '+4 Combative, violent, danger to staff',
        ],
        ordered=True)
    ts['richmond_ras_scale_goal'] = pd.Categorical(
        ts['richmond_ras_scale_goal'],
        categories=[
            '-5 Unarousable, no response to voice or physical stimulation'
            '-4 Deep sedation, no response to voice, but movement or eye opening to physical stimulation',
            '-3 Moderate sedation, movement or eye opening; No eye contact',
            '-2 Light sedation, briefly awakens to voice (eye opening/contact) < 10 sec',
            '-1 Awakens to voice (eye opening/contact) > 10 sec',
            ' 0  Alert and calm',
            '+1 Anxious, apprehensive, but not aggressive',
            '+2 Frequent nonpurposeful movement, fights ventilator',
            '+3 Pulls or removes tube(s) or catheter(s); aggressive',
            '+4 Combative, violent, danger to staff',
        ],
        ordered=True)
    ts['gcs_eyes'] = pd.Categorical(
        ts['gcs_eyes'],
        categories=[1.0, 2.0, 3.0, 4.0],
        ordered=True)
    ts['gcs_motor'] = pd.Categorical(
        ts['gcs_motor'],
        categories=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ordered=True)
    ts['gcs'] = pd.Categorical(
        ts['gcs'],
        categories=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        ordered=True)
    ts['gcs_verbal'] = pd.Categorical(
        ts['gcs_verbal'],
        categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ordered=True)
    ts['strength_l_arm'] = pd.Categorical(
        ts['strength_l_arm'],
        categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ordered=True)
    ts['strength_l_leg'] = pd.Categorical(
        ts['strength_l_leg'],
        categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ordered=True)
    ts['strength_r_arm'] = pd.Categorical(
        ts['strength_r_arm'],
        categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ordered=True)
    ts['strength_r_leg'] = pd.Categorical(
        ts['strength_r_leg'],
        categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ordered=True)

    ts = _encode_categories(ts, ts.select_dtypes(include=['category']).columns.tolist())
    ts = _encode_time(ts, ts.select_dtypes(include=['datetime']).columns.tolist())

    # adding `step` to index will explode memory with empty values
    ts.set_index(['stay_id'], inplace=True)
    ts.index.names = ['sample']
    ts.columns.name = 'feature'

    return ts.to_xarray()


def time_series_state(db: MimicDb, criteria: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    criteria_base = {
        'time_unit': 'hour',
        'time_step': '4'
    }
    criteria_final = prepare_criteria(criteria_base, criteria)

    ts = db.execute(MimicSql.TIME_SERIES_STATE, criteria_final, dtype={
        'stay_id': 'int32',
        'step': 'int16',
        # General
        'time_at': 'datetime64[ns]',
        'weight': 'float32',
        # Routine Vital Signs
        'diastolic_blood_pressure': 'float32',
        'ectopy_frequency': 'category',
        'ectopy_frequency_secondary': 'category',
        'ectopy_type': 'category',
        'ectopy_type_secondary': 'category',
        'heart_rate': 'float32',
        'heart_rhythm': 'category',
        'mean_blood_pressure': 'float32',
        'systolic_blood_pressure': 'float32',
        'temperature': 'float32',
        'temperature_site': 'category',
        # Chemistry
        'albumin': 'float32',
        'aniongap': 'float32',
        'bicarbonate': 'float32',
        'blood_urea_nitrogen': 'float32',
        'c_reactive_protein': 'float32',
        'calcium': 'float32',
        'chloride': 'float32',
        'creatinine': 'float32',
        'globulin': 'float32',
        'glucose': 'float32',
        'magnesium': 'float32',
        'phosphate': 'float32',
        'potassium': 'float32',
        'total_protein': 'float32',
        'sodium': 'float32',
        'ph_urine': 'float32',  # urinalysis
        # Chemistry | Enzymes
        'alanine_aminotransferase': 'float32',
        'alkaline_phosphatase': 'float32',
        'amylase': 'float32',
        'asparate_aminotransferase': 'float32',
        'bilirubin_direct': 'float32',
        'bilirubin_indirect': 'float32',
        'bilirubin_total': 'float32',
        'creatinine_kinase': 'float32',
        'gamma_glutamyltransferase': 'float32',
        'lactate_dehydrogenase': 'float32',
        # Chemistry | Cardiac Markers
        'creatinine_kinase_mb_isoenzyme': 'float32',
        'n_terminal_pro_hormone_bnp': 'float32',
        'troponin_t': 'float32',
        # Blood Gas
        'alveolar_arterial_gradient': 'float32',
        'arterial_o2_saturation': 'float32',
        'base_excess': 'float32',
        'total_co2': 'float32',
        'carboxyhemoglobin': 'float32',
        'free_calcium': 'float32',
        'lactate': 'float32',
        'methemoglobin': 'float32',
        'partial_pressure_of_co2': 'float32',
        'partial_pressure_of_o2': 'float32',
        'blood_gas_temperature': 'float32',
        'ph_blood': 'float32',
        # Hematology | Complete Blood Count
        'hematocrit': 'float32',
        'hemoglobin': 'float32',
        'mean_corpuscular_hemoglobin': 'float32',
        'mean_corpuscular_hemoglobin_concentration': 'float32',
        'mean_corpuscular_volume': 'float32',
        'platelet_count': 'float32',
        'red_blood_cells': 'float32',
        'red_cell_distribution_width': 'float32',
        'white_blood_cells': 'float32',
        # Hematology | Blood Differential
        'basophils_abs': 'float32',
        'eosinophils_abs': 'float32',
        'lymphocytes_abs': 'float32',
        'monocytes_abs': 'float32',
        'neutrophils_abs': 'float32',
        'atypical_lymphocytes': 'float32',
        'bands': 'float32',
        'basophils': 'float32',
        'eosinophils': 'float32',
        'immature_granulocytes': 'float32',
        'lymphocytes': 'float32',
        'metamyelocytes': 'float32',
        'monocytes': 'float32',
        'neutrophils': 'float32',
        'nucleated_red_cells': 'float32',
        # Hematology | Coagulation
        'd_dimer': 'float32',
        'fibrinogen': 'float32',
        'international_normalized_ratio': 'float32',
        'partial_thromboplastin_time': 'float32',
        'prothrombin_time': 'float32',
        'thrombin': 'float32',
        # Hemodynamics
        'intra_cranial_pressure': 'float32',
        # Output
        'urine_output': 'float32',
        # Respiratory
        'peripheral_o2_saturation': 'float32',
        # Respiratory | Ventilator Setting
        'flow_rate': 'float32',
        'fraction_inspired_o2': 'float32',
        'minute_volume': 'float32',
        'peep': 'float32',
        'plateau_pressure': 'float32',
        'respiratory_rate': 'float32',
        'respiratory_rate_set': 'float32',
        'respiratory_rate_spontaneous': 'float32',
        'tidal_volume_observed': 'float32',
        'tidal_volume_set': 'float32',
        'tidal_volume_spontaneous': 'float32',
        'ventilator_mode': 'category',
        'ventilator_mode_hamilton': 'category',
        'ventilator_type': 'category',
        # Respiratory | Oxygen Delivery
        'o2_delivery_device_1': 'category',
        'o2_delivery_device_2': 'category',
        'o2_delivery_device_3': 'category',
        'o2_delivery_device_4': 'category',
        'o2_flow': 'float32',
        'o2_flow_additional': 'float32',
        # Neurological
        'gcs_eyes': 'category',
        'gcs_motor': 'category',
        'gcs': 'category',
        'gcs_unable': 'category',
        'gcs_verbal': 'category',
        'strength_l_arm': 'category',
        'strength_l_leg': 'category',
        'strength_r_arm': 'category',
        'strength_r_leg': 'category',
        # Cardiovascular (Pulses)
        'capillary_refill_rate_l': 'category',
        'capillary_refill_rate_r': 'category',
        # Skin - Assessment
        'braden_activity': 'category',
        'braden_friction': 'category',
        'braden_mobility': 'category',
        'braden_moisture': 'category',
        'braden_nutrition': 'category',
        'braden_sensory_perception': 'category',
        'skin_temperature': 'category',
        # Alarms
        'alarms_on': 'category',
        'heart_rate_alarm_high': 'float32',
        'heart_rate_alarm_low': 'float32',
        'blood_pressure_alarm_high': 'float32',
        'blood_pressure_alarm_low': 'float32',
        'o2_saturation_desat_limit': 'float32',
        'o2_saturation_alarm_high': 'float32',
        'o2_saturation_alarm_low': 'float32',
        'parameters_checked': 'category',
        'respiratory_rate_alarm_high': 'float32',
        'respiratory_rate_alarm_low': 'float32',
        'st_segment_monitoring_on': 'category',
        # Restraint/Support Systems
        'ambulatory_aid': 'category',
        'gait_transferring': 'category',
        'high_risk_gt_51_interventions': 'category',
        'history_of_falling_3m': 'category',
        'iv_saline_lock': 'category',
        'mental_status': 'category',
        'secondary_diagnosis': 'category',
        # Pain/Sedation
        'pain_level': 'category',
        'pain_level_response': 'category',
        'richmond_ras_scale': 'category',
        'richmond_ras_scale_goal': 'category',
        # Access Lines - Peripheral
        'gauge_18_dressing_occlusive': 'category',
        'gauge_18_placed_in_outside_facility': 'category',
        'gauge_20_dressing_occlusive': 'category',
        'gauge_20_placed_in_outside_facility': 'category',
        'gauge_20_placed_in_the_field': 'category',
        # Treatments
        'eye_care': 'category'
    }, chunksize=1000000)
    ts.set_index(['stay_id', 'step'], inplace=True)
    return ts


def prepare_criteria(criteria_base: Dict[str, str], criteria: Optional[Dict[str, str]] = None):
    criteria_final = {**criteria_base, ** (criteria if criteria is not None else {})}
    return criteria_final


def signals_event_count(db: MimicDb, criteria: Optional[Dict[str, str]] = None) -> Dataset:
    ts = time_series_state_count(db, criteria).add_prefix('n_').reset_index()

    # adding `step` to index will explode memory with empty valuses
    ts.set_index(['stay_id'], inplace=True)
    ts.index.names = ['sample']
    ts.columns.name = 'feature'
    return ts.to_xarray()


def time_series_state_count(db: MimicDb, criteria: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    criteria_base = {
        'time_unit': 'hour',
        'time_step': '4'
    }
    criteria_final = prepare_criteria(criteria_base, criteria)

    ts = db.execute(MimicSql.TIME_SERIES_STATE_COUNT, criteria_final, chunksize=1000000)
    ts.set_index(['stay_id', 'step'], inplace=True)
    ts.fillna(0, inplace=True)
    return ts.astype('int16')
