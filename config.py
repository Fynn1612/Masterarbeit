import os
from dotenv import load_dotenv

# Lade .env
load_dotenv()

# Hole Pfade aus Environment-Variablen
DATA_PATH = os.getenv('DATA_PATH')
DE_PREDICTION_PATH = os.getenv('DE_PREDICTION_PATH')
DE_RESULTS_PATH = os.getenv('DE_RESULTS_PATH')


# Validierung
if not all([DATA_PATH, DE_PREDICTION_PATH, DE_RESULTS_PATH]):
    raise ValueError("Bitte .env-Datei mit DATA_PATH, DE_PREDICTION_PATH, DE_RESULTS_PATH erstellen!")