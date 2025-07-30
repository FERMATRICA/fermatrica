"""
Objective functions, wrappers and other components for model outer layer optimization.

Both global and local algorithms are supported. As for now two derivative-free algos are included:
1. COBYLA constrained optimization by linear approximations (local)
2. PyGad genetic algorithm (global)

However, FERMATRICA architecture allows fast and simple adding new algorithms, and some
algos could be added later.

Derivative-free approach allows optimising without calculating (analytical) gradient what could be
very complex and time-consuming. However, some derivative algo (e.g. GS) could be added later
at least for some most popular transformations.

Some "extra" imports in this file better to be preserved to be accessible for parallel computations
"""

import copy
import datetime
import logging
import math
import pandas as pd
import numpy as np

from scipy.optimize import minimize

from fermatrica.model.transform import transform
from fermatrica.model.predict import fit_predict
import fermatrica.evaluation.scoring as scr
from fermatrica.model.model import Model


class ObjectiveFunction:
    """
    Objective function generator / fabric. Variables required by model to run could be assigned to
    the closure one time instead of passing every iteration. Works fine in single-process environment
    only, so to be used for local algorithms mostly.
    """

    def __init__(self,
                 ds: pd.DataFrame,
                 model: "Model",
                 revert_score_sign: bool = False,
                 error_score: float = 1e+12,
                 verbose: bool = True):
        """
        :param ds: dataset
        :param model: Model object
        :param revert_score_sign: if algo minimizes score, and the score is the bigger, the better; or vice versa
        :param error_score: extremely big (or small) value to be used as score if fit_predict returns None (error)
        :param verbose: print diagnostic / progress info
        """

        # set optimising / data environment in class because some algos cannot read additional arguments

        self.ds = ds
        self._prepare_model(model)
        self._prepare_mask()
        self.revert_score_sign = revert_score_sign
        self.error_score = error_score
        self.verbose = verbose
        self.iter_idx = 1

    def _prepare_model(self
                       , model: "Model"):
        """
        Prepare main model object for optimising

        :param model: Model object
        :return:
        """

        model = copy.deepcopy(model)

        # Remove extra data to make deepcopy model_wrk creation more preformance efficient

        model.obj.models = {}

        # Replace extremely small numbers close to machine precision as zeroes

        mask = (model.conf.params['type'].isin(['float64', 'int'])) & \
               (abs(pd.to_numeric(model.conf.params['value'], errors='coerce')) < 1e-11)
        model.conf.params.loc[mask, 'value'] = 0

        # assign

        self.model = model

    def _prepare_mask(self):
        self.optim_mask = (self.model.conf.params['if_active'] == 1) & (self.model.conf.params['fixed'] == 0)

    def __call__(self,
                 params_cur: list | np.ndarray | pd.Series | None = None,
                 solution_idx=None) -> float | None:
        """
        Main logic. Actually objective function to be used in algos with environment variables as class attributes

        :param params_cur: current params vector
        :param solution_idx: not used explicitly
        :return: current score
        """

        if params_cur is not None:
            if np.isnan(params_cur).any():
                return self.error_score

            if isinstance(params_cur, pd.Series):
                params_cur = params_cur.values

        model_wrk = copy.deepcopy(self.model)

        if params_cur is not None:
            model_wrk.conf.params.loc[self.optim_mask, 'value'] = params_cur

        model_wrk = transform(ds=self.ds,
                             model=model_wrk,
                             set_start=True,
                             if_by_ref=True)

        pred, pred_raw, model_wrk = fit_predict(self.ds,
                                                model_wrk,
                                                if_fit=True,
                                                return_full=True)

        score = self._calculate_score(pred, model_wrk)

        del pred, pred_raw, model_wrk

        if self.verbose:
            self._verbose(score)

        return score

    def _calculate_score(self
                         , pred: list | tuple | np.ndarray | pd.Series | None
                         , model: "Model") -> float:
        """
        Get scoring

        :param pred: vector of predicted values
        :param model: Model object of the current iteration ("working")
        :return:
        """

        if pred is None:
            return self.error_score

        score = scr.scoring(self.ds[self.ds['listed'] == 2], pred, model)

        if math.isinf(score):
            score = self.error_score

        if self.revert_score_sign and score != self.error_score:
            score = -score

        return score

    def _verbose(self
                 , score: float):
        """
        Verbose / log every 10th iteration

        :param score: current score
        :return: pass
        """

        if self.iter_idx % 10 == 0:
            score_print = (round(score, 5)
                           if abs(score) < 500
                           else round(score, 0))
            timestamp = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
            print(f"Combined scoring : {score_print}; Step : {self.iter_idx}; Time : {timestamp}")

        self.iter_idx += 1


def objective_fun_args(params_cur: list | np.ndarray | pd.Series | None
                       , solution_idx
                       , args: dict):
    """
    Objective function. Flavour for parallel computing, so all args are in a dict and no non-local variables
    are used. Function generator / fabric is also not used because of the same reason.

    :param params_cur: current params vector
    :param solution_idx: not used explicitly
    :param args: arguments for model run packed in dictionary
    :return: current score
    """

    ds = args['ds']
    model = args['model']
    revert_score_sign = args['revert_score_sign']
    error_score = args['error_score']

    model_wrk = copy.deepcopy(model)

    # apply current values (from algo) to params frame (for predict function)

    if np.isnan(params_cur).any():
        return error_score

    if isinstance(params_cur, pd.Series):
        params_cur = params_cur.values

    if params_cur is not None:
        model_wrk.conf.params.loc[(model_wrk.conf.params['if_active'] == 1) & (model_wrk.conf.params['fixed'] == 0), 'value'] = params_cur

    # RHS transformation - not in predict function, so one can use predict w/o RHS transformation and
    #   save plenty of time when RHS is already transformed

    model_wrk = transform(ds=ds
                      , model=model_wrk
                      , set_start=True
                      , if_by_ref=True)

    # get full prediction

    pred, pred_raw, model_wrk = fit_predict(ds
                                        , model_wrk
                                        , if_fit=True
                                        , return_full=True)

    # get scoring

    if pred is None:
        score = None
    else:
        score = scr.scoring(ds[ds['listed'] == 2], pred, model_wrk)

    if (score is None) or (math.isinf(score)):
        score = error_score
    elif revert_score_sign:
        score = -score


    # return

    return score


class ObjectiveNG:

    def __init__(self
                 , param_names: list
                 , objective_fun):

        self.param_names = param_names
        self.objective_fun = objective_fun

    def __call__(self
                 , **params_dict):

        params = []

        for par in self.param_names:
            param_value = params_dict[par]
            params.append(param_value)

        score = self.objective_fun(params)

        return score


class ObjectiveRBF:

    def __init__(self
                 , param_train
                 , score_train
                 , ub
                 , lb
                 , iters_epoch
                 , ftol_abs
                 , sm_object):

        self.param_train = param_train
        self.score_train = score_train
        self.ub = ub
        self.lb = lb
        self.iters_epoch = iters_epoch
        self.ftol_abs = ftol_abs
        self.sm_object = sm_object

    def __call__(self
                 , **params_dict):

        x0 = self.param_train[np.argmin(self.score_train)] + np.random.normal(0, (self.ub - self.lb) * 0.01)
        x0 = np.clip(x0, self.lb, self.ub)

        res = minimize(lambda x: self.sm_object.predict_values(x.reshape(1, -1))[0],
                       x0,
                       method='L-BFGS-B',
                       options={
                           'maxiter': self.iters_epoch,
                           'ftol': self.ftol_abs,
                           'disp': False
                       },
                       bounds=[(l, u) for l, u in zip(self.lb, self.ub)])

        return res.x


"""
Objective callbacks
"""

class EarlyStopOptuna:
    """
    Early Stopping Callback for Optuna optimizers.
    More conditions could be added later
    """

    def __init__(self
                 , ftol_abs: float | None = None
                 , max_no_improvement: int | None = None
                 , if_min: bool = True):

        self.ftol_abs = ftol_abs
        self.max_no_improvement = max_no_improvement

        if if_min:
            self.last_best_score = float('inf')
        else:
            self.last_best_score = float('-inf')

        self.if_min = if_min
        self.no_improvement_count = 0

    def __call__(self
                 , study
                 , trial):

        current_score = study.best_value

        if self.ftol_abs is not None and self.max_no_improvement is not None:

            if not self.if_min:
                # for maximization, lower negative score is better
                improvement = current_score - self.last_best_score
            else:
                improvement = self.last_best_score - current_score

            if improvement >= self.ftol_abs:
                self.last_best_score = current_score
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            # check stopping condition
            if self.no_improvement_count >= self.max_no_improvement:
                study.stop()

        else:
            logging.warning("Early stop in optimisation was not fulfilled because conditions were not set")


class VerboseOptuna:
    """
    Optuna callback to print progress
    """

    def __init__(self
                 , verbose_level: int = 10):
        """

        :param verbose_level: output frequency (every N-th trial)
        """
        if verbose_level <= 0:
            raise ValueError("verbose_level should be positive number")

        self.verbose_level = int(verbose_level)

    def __call__(self, study, trial):

        if trial.state.is_finished() and ((trial.number + 1) % self.verbose_level == 0):
            print('Combined scoring : ' + str(trial.value) + ' ; step : ' +
                  str(trial.number + 1) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))


class VerboseNG:
    """
    Nevergrad callback to print progress
    """
    def __init__(self
                 , verbose_level: int = 10):

        self.verbose_level = verbose_level
        self.iter_idx = 1

    def __call__(self
                 , optimizer
                 , parametrization
                 , loss: float
                 , *args
                 , **kwargs):

        if self.iter_idx % self.verbose_level == 0:

            score_print = round(loss, 5) if abs(loss) < 500 else round(loss, 0)
            timestamp = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")

            print(f"Combined scoring : {score_print}; step : {self.iter_idx} : {timestamp}")

        self.iter_idx += 1


"""
Other helpers
"""


def _save_model(model_iter: "Model"
                , model: "Model"
                , ds: pd.DataFrame
                , path: str | bool = False):

    """
    Re-run model and save
    """

    try:

        model_iter.obj.transform_lhs_fn = model.obj.transform_lhs_fn

        model_iter = transform(ds=ds
                               , model=model_iter
                               , set_start=False
                               , if_by_ref=True)

        pred, pred_raw, model_iter = fit_predict(ds=ds
                                                 , model=model_iter
                                                 , if_fit=True
                                                 , return_full=True)

        if isinstance(path, bool) and path is True:
            model_iter.save(ds=ds)

        elif isinstance(path, str):
            model_iter.save(ds=ds, path=path)

    except Exception as e:
        logging.error("Failed to save model: " + str(e))
