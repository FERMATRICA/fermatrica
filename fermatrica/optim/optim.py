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


from contextlib import suppress
import copy
import datetime
import ipyparallel.serialize.codeutil
import logging
import math
import multiprocessing
import pandas as pd
import numpy as np
import os
import pickle
import typing
import warnings

import nlopt
import fermatrica.optim.pygad as pygad

import fermatrica
from fermatrica.basics.basics import fermatrica_error
from fermatrica.model.transform import transform
from fermatrica.model.predict import fit_predict
import fermatrica.evaluation.scoring as scr
from fermatrica.model.model import Model


def objective_fun_generate(ds: pd.DataFrame
                           , model: "Model"
                           , revert_score_sign: bool = False
                           , error_score: float = 1e+12
                           , verbose: bool = True):
    """
    Objective function generator / fabric. Variables required by model to run could be assigned to
    the closure one time instead of passing every iteration. Works fine in single-process environment
    only, so to be used for local algorithms mostly.

    :param ds: dataset
    :param model: Model object
    :param revert_score_sign: if algo minimizes score, and the score is the bigger, the better; or vice versa
    :param error_score: extremely big (or small) value to be used as score if fit_predict returns None (error)
    :param verbose: print diagnostic / progress info
    :return: generated objective function
    """

    # set optimising / data environment in closure because some algos cannot read additional arguments

    iter_idx = 1

    model.conf.params.loc[(model.conf.params['type'].isin(['float64', 'int'])) &
                          (abs(pd.to_numeric(model.conf.params['value'], errors='coerce')) < 1e-11), 'value'] = 0

    def objective_fun(params_cur: list | np.ndarray | pd.Series | None = None
                      , solution_idx=None):
        """
        Objective function with environment variables from closure.

        :param params_cur: current params vector
        :param solution_idx: not used explicitly
        :return: current score
        """

        # use `nonlocal` to access data stored in closure

        nonlocal model
        nonlocal ds
        nonlocal revert_score_sign
        nonlocal verbose
        nonlocal error_score
        nonlocal iter_idx

        # despite `deepcopy()` original model is overwritten at least sometimes
        # when just model = copy.deepcopy(model). So new variable is introduced
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

        # print info

        if verbose:
            if iter_idx % 10 == 0:

                score_print = round(score, 5) if abs(score) < 500 else round(score, 0)
                print('Combined scoring : ' + str(score_print) + ' ; step : ' + str(iter_idx) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            iter_idx += 1

        # return

        return score

    return objective_fun


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


def optimize_local_cobyla(ds: pd.DataFrame
                          , model: "Model"
                          , revert_score_sign: bool = False
                          , verbose: bool = True
                          , epochs: int = 3
                          , iters_epoch: int = 300
                          , error_score: float = 1e+12
                          , ftol_abs: float = .01):
    """
    Wrapper for COBYLA constrained optimization by local approximations algorithm (local). COBYLA
    is derivative-free, so no analytical gradient is required.

    COBYLA minimizes score, so be careful to revert score if it is designed as maximizer before use
    or with argument `revert_score_sign=True`.

    Epochs are required to shuffle params a bit and to help algo get out from local optimum (sometimes)
    and to speed up calculation.

    Early stop threshold is defined as minimum absolute score gain per iteration. However, algo doesn't
    respect it directly, so don't be upset to see it working with much lesser gain per iteration.

    Local algorithms could be sensitive to starting (initial) values and to the number of optimising
    params. If params number is big enough (say, dozens) it is better to use it chunks, tuning
    one portion of params after another. Params for every chunk should be chosen wisely and not
    just first 10 or 20 params, them second 10 or 20 etc.

    Nlopt COBYLA implementation is preferred before `scipy.optimize.minimize(method='COBYLA')` because
    latter doesn't respect constraints accurately.

    However, Nlopt COBYLA has its own annoying issue. If improvement is (much?) below machine precision,
    it stops with error. We have some thoughts about workarounds to be implemented later, but as for now
    be aware starting values are not to close to the borders, e.g. 0.9999999999999 when border is 1.0.
    Move starting value a little further, e.g. to .95, and in most times it will be enough.

    :param ds: dataset
    :param model: Model object
    :param revert_score_sign: if the score is the bigger, the better; this algo minimizes score
    :param verbose: print diagnostic or progress information
    :param epochs: number of epochs
    :param iters_epoch: number of objective function calculations per epoch
    :param error_score: extremely big value to be used as score if fit_predict returns None (error)
    :param ftol_abs: early stop threshold: minimum absolute score gain per iteration, see description
    :return: tuned Model
    """

    params_subset = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]

    for i in range(epochs):

        if_error = False

        # rebuild objective function every epoch to update model.conf
        objective_fun = objective_fun_generate(ds=ds
                                               , model=model
                                               , revert_score_sign=revert_score_sign
                                               , error_score=error_score
                                               , verbose=verbose)

        start_values = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]['value'].values.tolist()

        opt_obj = nlopt.opt(nlopt.LN_COBYLA, len(start_values))

        opt_obj.set_min_objective(objective_fun)
        opt_obj.set_lower_bounds(params_subset['lower'].values)
        opt_obj.set_upper_bounds(params_subset['upper'].values)

        opt_obj.set_ftol_abs(ftol_abs)
        opt_obj.set_maxeval(iters_epoch)
        opt_obj.set_maxtime(0)

        try:
            rtrn = opt_obj.optimize(start_values)
        except (nlopt.RoundoffLimited, ValueError):
            rtrn = start_values
            if_error = True
            logging.warning("Epoche terminated unsuccessfully due to ROUNDOFF error")

        print('Epoche ' + str(i + 1) + ' finished')

        if (isinstance(rtrn, np.ndarray)) or (isinstance(rtrn, list)):
            model.conf.params.loc[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0), 'value'] = rtrn
        else:
            logging.warning("Epoche did not terminate successfully")

        model.conf.params.loc[model.conf.params['value'] > model.conf.params['upper'], 'value'] =\
            model.conf.params[model.conf.params['value'] > model.conf.params['upper']]['upper']
        model.conf.params.loc[model.conf.params['value'] < model.conf.params['lower'], 'value'] =\
            model.conf.params[model.conf.params['value'] < model.conf.params['lower']]['lower']

        if if_error:
            break

    return model


def optimize_global_ga(ds: pd.DataFrame
                       , model: "Model"
                       , revert_score_sign: bool = False
                       , verbose: bool = True
                       , epochs: int = 1
                       , iters_epoch: int = 300
                       , pop_size: int = 50
                       , pmutation: float = .1
                       , max_no_improvement: int = 20
                       , ftol_abs: float = 0
                       , error_score: float = -1e+12
                       , cores: int = 4
                       , save_epoch: bool | str = True):
    """


    :param ds: dataset
    :param model: Model object
    :param revert_score_sign: if the score is the smaller, the better; this algo maximizes score
    :param verbose: print diagnostic or progress information
    :param epochs: number of epochs; every epoch is independent form others and return its own score and tuned
        params vector
    :param iters_epoch: number of iterations (not objective function calculation) per epoch
    :param pop_size: GA param: population size
    :param pmutation: GA param: permutations share
    :param max_no_improvement: maximum number of iterations as early stop threshold
    :param ftol_abs: early stop threshold: minimum absolute score gain per `max_no_improvement` number of iteration,
        defaults to 0
    :param error_score: extremely small value to be used as score if fit_predict returns None (error)
    :param cores: processor cores to be used in parallel computing
    :param save_epoch: if every epoch to be saved or path to the folder to save tuned models
    :return: tuned Model of the last epoch
    """

    # preparations and checks

    params_subset = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]

    start_values = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)][
        'value'].values.tolist()

    if verbose:
        def callback_gen(ga_instance):
            print('Combined scoring : ' + str(ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]) + ' ; step : ' +
                  str(ga_instance.generations_completed) + ' : ' + datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
    else:
        callback_gen = None

    keep_elitism = int(round(pop_size * .1))
    if keep_elitism < 1:
        keep_elitism = 1

    if cores > (multiprocessing.cpu_count() - 1):
        cores = (multiprocessing.cpu_count() - 1)
        logging.warning('Cores number for parallel computing is set too high for this computer. ' +
                        'Reset to ' + str(cores))

    # run algo

    for i in range(epochs):

        # keep basic model unaltered to make epochs independent
        model_iter = copy.deepcopy(model)

        # destroy callable objects for multiprocessing
        model_iter.obj.transform_lhs_fn = None

        lwr = params_subset['lower'].values
        upr = params_subset['upper'].values

        lwr = np.nan_to_num(lwr, nan=-1e+8)
        upr = np.nan_to_num(upr, nan=1e+8)

        bndr = list(map(list, zip(lwr, upr)))
        bndr = [{'low': x[0], 'high': x[1]} for x in bndr]

        gad_model = pygad.GA(num_generations=iters_epoch
                             , num_parents_mating=2
                             , fitness_func=objective_fun_args
                             , args={'ds': ds
                                     , 'model': model_iter
                                     , 'revert_score_sign': revert_score_sign
                                     , 'error_score': error_score
                                     }
                             , num_genes=len(start_values)
                             , sol_per_pop=pop_size
                             , parent_selection_type="sss"
                             , keep_elitism=keep_elitism
                             , gene_space=bndr
                             , crossover_type="single_point"
                             , mutation_type="random"
                             , mutation_probability=pmutation
                             , stop_criteria='saturate_'+str(int(max_no_improvement))
                             , ftol_abs=ftol_abs
                             , on_generation=callback_gen
                             , suppress_warnings=True
                             , parallel_processing=['process', cores]
                             , delay_after_gen=0
                             , save_best_solutions=True
                             )

        try:
            gad_model.run()
            rtrn = gad_model.best_solution()[0]
            gad_model.close_parallel()
        except Exception as e:
            rtrn = start_values
            logging.warning("Genetic Algorithm error")
            if hasattr(e, 'message'):
                print(e.message)

        print('Epoche ' + str(i + 1) + ' finished')

        if (isinstance(rtrn, np.ndarray)) or (isinstance(rtrn, list)):
            model_iter.conf.params.loc[(model_iter.conf.params['if_active'] == 1) & (model_iter.conf.params['fixed'] == 0), 'value'] = rtrn
        else:
            logging.warning("Epoche did not terminate successfully")

        model_iter.conf.params.loc[model_iter.conf.params['value'] > model_iter.conf.params['upper'], 'value'] =\
            model_iter.conf.params[model_iter.conf.params['value'] > model_iter.conf.params['upper']]['upper']
        model_iter.conf.params.loc[model_iter.conf.params['value'] < model_iter.conf.params['lower'], 'value'] =\
            model_iter.conf.params[model_iter.conf.params['value'] < model_iter.conf.params['lower']]['lower']

        if save_epoch:

            model_iter.obj.transform_lhs_fn = model.obj.transform_lhs_fn

            model_iter = transform(ds=ds
                                   , model=model_iter
                                   , set_start=False
                                   , if_by_ref=True)

            pred, pred_raw, model_iter = fit_predict(ds=ds
                                                     , model=model_iter
                                                     , if_fit=True
                                                     , return_full=True)

            if isinstance(save_epoch, bool) and save_epoch is True:
                model_iter.save(ds=ds)

            elif isinstance(save_epoch, str):
                model_iter.save(ds=ds, path=save_epoch)

    return model_iter

