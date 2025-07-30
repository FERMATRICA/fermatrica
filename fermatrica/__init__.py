"""
FERMATRICA

Core of the FERMATRICA econometrics framework: define model, build, run, evaluate and optimize
"""


import fermatrica.basics
import fermatrica.model
import fermatrica.evaluation
import fermatrica.optim

from fermatrica.basics.basics import fermatrica_error, FermatricaError, params_to_dict
from fermatrica.model.model import Model, model_load_ext
from fermatrica.model.model_conf import ModelConf
from fermatrica.model.model_obj import ModelObj, prepickle as model_obj_prepickle
from fermatrica.model.transform import transform
from fermatrica.model.predict import fit_predict, predict_ext
import fermatrica.evaluation.metrics as metrics
from fermatrica.evaluation.scoring import scoring
from fermatrica.optim.locals import optimize_local_cobyla, optimize_local_bobyqa, optimize_local_sbplx, \
    optimize_local_rbf, optimize_local_ncma
from fermatrica.optim.globals import optimize_global_ga, optimize_global_tpe
from fermatrica.optim.globals_deap import optimize_global_de
