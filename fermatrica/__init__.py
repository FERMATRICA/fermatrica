"""
FERMATRICA

Core of the FERMATRICA econometrics framework: define model, build, run, evaluate and optimize
"""


import fermatrica.basics
import fermatrica.model
import fermatrica.evaluation
import fermatrica.optim

from fermatrica.basics.basics import fermatrica_error, FermatricaError
from fermatrica.model.model import Model, model_load_ext
from fermatrica.model.model_conf import ModelConf
from fermatrica.model.model_obj import ModelObj, prepickle as model_obj_prepickle
from fermatrica.model.transform import transform
from fermatrica.model.predict import fit_predict, predict_ext
import fermatrica.evaluation.metrics as metrics
from fermatrica.evaluation.scoring import scoring
from fermatrica.optim.optim import optimize_local_cobyla, optimize_global_ga
