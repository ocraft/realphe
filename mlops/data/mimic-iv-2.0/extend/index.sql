SET search_path TO mimiciv_hosp;

DROP INDEX IF EXISTS labevents_idx03;
CREATE INDEX labevents_idx03
  ON labevents (itemid);

DROP INDEX IF EXISTS services_idx01;
CREATE INDEX services_idx01
  ON services (hadm_id, transfertime);

SET search_path TO mimiciv_icu;

DROP INDEX IF EXISTS inputevents_idx03;
CREATE INDEX inputevents_idx03
  ON inputevents (stay_id);

DROP INDEX IF EXISTS chartevents_idx02;
CREATE INDEX chartevents_idx02
  ON chartevents (stay_id);

DROP INDEX IF EXISTS datetimeevents_idx03;
CREATE INDEX datetimeevents_idx03
  ON datetimeevents (itemid);

DROP INDEX IF EXISTS datetimeevents_idx04;
CREATE INDEX datetimeevents_idx04
  ON datetimeevents (stay_id);

DROP INDEX IF EXISTS ingredientevents_idx01;
CREATE INDEX ingredientevents_idx01
  ON ingredientevents (itemid);

DROP INDEX IF EXISTS ingredientevents_idx02;
CREATE INDEX ingredientevents_idx02
  ON ingredientevents (stay_id);

DROP INDEX IF EXISTS inputevents_idx04;
CREATE INDEX inputevents_idx04
  ON inputevents (itemid);

DROP INDEX IF EXISTS outputevents_idx02;
CREATE INDEX outputevents_idx02
  ON outputevents (itemid);

DROP INDEX IF EXISTS outputevents_idx03;
CREATE INDEX outputevents_idx03
  ON outputevents (stay_id);

DROP INDEX IF EXISTS procedureevents_idx03;
CREATE INDEX procedureevents_idx03
  ON procedureevents (itemid);

DROP INDEX IF EXISTS procedureevents_idx04;
CREATE INDEX procedureevents_idx04
  ON procedureevents (stay_id);

SET search_path TO public;

DROP INDEX IF EXISTS icd10_to_icd9_idx01;
CREATE INDEX icd10_to_icd9_idx01
  ON icd10_to_icd9 (icd10_code);

DROP INDEX IF EXISTS icd9_to_ccs_idx01;
CREATE INDEX icd9_to_ccs_idx01
  ON icd9_to_ccs (icd9_code);

DROP INDEX IF EXISTS icd_to_phecode_idx01;
CREATE INDEX icd_to_phecode_idx01
  ON icd_to_phecode (icd);

DROP INDEX IF EXISTS icd_to_phecode_idx02;
CREATE INDEX icd_to_phecode_idx02
  ON icd_to_phecode (icd,flag);

SET search_path TO mimiciv_derived;

DROP INDEX IF EXISTS charts_idx01;
CREATE INDEX charts_idx01
  ON charts (stay_id);
