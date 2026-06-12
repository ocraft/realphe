SET search_path TO mimiciv_derived;

DROP INDEX IF EXISTS vitalsign_idx01;
CREATE INDEX vitalsign_idx01
  ON vitalsign (stay_id);

DROP INDEX IF EXISTS chemistry_idx01;
CREATE INDEX chemistry_idx01
  ON chemistry (hadm_id);

DROP INDEX IF EXISTS cardiac_marker_idx01;
CREATE INDEX cardiac_marker_idx01
  ON cardiac_marker (hadm_id);

DROP INDEX IF EXISTS cardiac_marker_idx02;
CREATE INDEX cardiac_marker_idx02
  ON cardiac_marker (specimen_id);

DROP INDEX IF EXISTS sofa_idx01;
CREATE INDEX sofa_idx01
  ON sofa (stay_id);

DROP INDEX IF EXISTS icustay_detail_idx01;
CREATE INDEX icustay_detail_idx01
  ON icustay_detail (stay_id);

DROP INDEX IF EXISTS icustay_detail_idx02;
CREATE INDEX icustay_detail_idx02
  ON icustay_detail (subject_id);

SET search_path TO mimiciv_hosp;

DROP INDEX IF EXISTS labevents_idx04;
CREATE INDEX labevents_idx04
  ON labevents (hadm_id);
