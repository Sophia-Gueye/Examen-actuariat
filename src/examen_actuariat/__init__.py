"""
Package examen actuariat.

Ce package contient des modules pour :
- Chargement des données (data_loading)
- Préprocessing des données (data_processing)
- Analyse exploratoire (exploration)
- Extraction de features (features)
- Modélisation (models)
- Évaluation (evaluation)
"""

from . import data_loading
from . import data_processing
from . import exploration
from . import features
from . import models
from . import evaluation

_all_ = [
    "data_loading",
    "data_processing", 
    "exploration",
    "features",
    "models",
    "evaluation"
]