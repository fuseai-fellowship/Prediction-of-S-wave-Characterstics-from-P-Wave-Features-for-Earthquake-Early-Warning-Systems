import os

# output artifact defaults
ARTIFACT_DIR = os.environ.get("EEW_ARTIFACTS_DIR", "artifacts")
MODEL_NAME = os.environ.get("EEW_MODEL_NAME", "xgb_eew_final.joblib")
PREPROC_NAME = os.environ.get("EEW_PREPROC_NAME", "preproc_objects.joblib")

# default feature list name; training used these names
P_WAVE_FEATURES = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]