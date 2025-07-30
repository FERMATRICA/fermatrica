"""
Wrappers for model outer layer optimization (global algos).

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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import pandas as pd

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from fermatrica.model.model import Model
from fermatrica.optim import pygad as pygad
from fermatrica.optim.objective import (objective_fun_args, ObjectiveFunction, EarlyStopOptuna, VerboseOptuna,
                                        _save_model)


optuna.logging.set_verbosity(optuna.logging.WARNING)


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

    if params_subset.empty:
        logging.warning("No active non-fixed parameters to optimize")
        return model

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

    for epoch in range(epochs):

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

        print('Epoche ' + str(epoch + 1) + ' finished')

        if (isinstance(rtrn, np.ndarray)) or (isinstance(rtrn, list)):
            model_iter.conf.params.loc[(model_iter.conf.params['if_active'] == 1) & (model_iter.conf.params['fixed'] == 0), 'value'] = rtrn
        else:
            logging.warning("Epoche did not terminate successfully")

        optim_mask = (model_iter.conf.params['if_active'] == 1) & (model_iter.conf.params['fixed'] == 0)
        upper_mask = (model_iter.conf.params['value'] > model_iter.conf.params['upper']) & optim_mask
        model_iter.conf.params.loc[upper_mask, 'value'] = model_iter.conf.params.loc[upper_mask, 'upper']
        lower_mask = (model_iter.conf.params['value'] < model_iter.conf.params['lower']) & optim_mask
        model_iter.conf.params.loc[lower_mask, 'value'] = model_iter.conf.params.loc[lower_mask, 'lower']

        if save_epoch:
            _save_model(model_iter, model, ds, save_epoch)

    return model_iter


def optimize_global_tpe(ds: pd.DataFrame,
                 model: "Model",
                 revert_score_sign: bool = False,
                 verbose: bool = True,
                 epochs: int = 3,
                 iters_epoch: int = 300,
                 error_score: float = 1e+12,
                 ftol_abs: float = .01,
                 max_no_improvement: int = 5,
                 approach: str = 'void',
                 save_epoch: bool | str = True):
    """
    Wrapper for TPE Bayesian optimization algorithm using Optuna with epoch-based execution.
    TPE is a derivative-free global optimizer that maintains search history between epochs.
    TPE minimizes score, so be careful to revert score if it is designed as maximizer before use
    or with argument `revert_score_sign=True`.

    :param ds: dataset
    :param model: Model object
    :param revert_score_sign: if the score is the bigger, the better; this algo minimizes score
    :param verbose: print diagnostic or progress information
    :param epochs: number of epochs; each epoch is independent and returns its own score and tuned params
    :param iters_epoch: number of trials per epoch
    :param error_score: extremely big value to be used as score if fit_predict returns None (error)
    :param max_no_improvement: number of trials with no significant improvement to trigger early stopping
    :param approach: 'void', 'init', 'sequent', 'init_sequent'
    :param ftol_abs: early stop threshold (unused in current implementation)
    :param save_epoch: if every epoch to be saved or path to the folder to save tuned models
    :return: tuned Model of the last epoch
    """

    # set up settings

    params_subset = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]

    if params_subset.empty:
        logging.warning("No active non-fixed parameters to optimize")
        return model

    params_list = list(zip(
        params_subset['value'].values
        , np.nan_to_num(params_subset['lower'].values, nan=-1e8)
        , np.nan_to_num(params_subset['upper'].values, nan=1e8)
        , params_subset['type'].values
    ))

    # main epoch loop

    for epoch in range(epochs):

        # recreate study for each epoch

        study = optuna.create_study(
            sampler=TPESampler(),
            direction='minimize',
            pruner=SuccessiveHalvingPruner(),  # trial-level pruning
            storage=None,
            study_name="tpe_optimization",
            load_if_exists=True
        )

        # create new objective function instance for each epoch

        model_iter = copy.deepcopy(model)
        model_iter.obj.transform_lhs_fn = None

        objective_fun = ObjectiveFunction(
            ds=ds,
            model=model_iter,
            revert_score_sign=revert_score_sign,
            error_score=error_score,
            verbose=False
        )

        # optuna wrapper over classic optimising interface

        def optuna_objective(trial):

            # generate parameters for each dimension
            params = []

            for ind, par in enumerate(params_list):
                if par[3] in ['int', 'int64', 'int32']:
                    param_value = trial.suggest_int('p_' + str(ind), par[1], par[2])
                else:
                    param_value = trial.suggest_float('p_' + str(ind), par[1], par[2])
                params.append(param_value)

            # run
            score = objective_fun(params)

            return score

        # add warm start if we have previous trials of initial values are used

        if approach == 'init' or (approach == 'init_sequent' and epoch == 0):
            try:
                # set initial values
                warm_start_params = {"p_" + str(idx): value[0] for idx, value in enumerate(params_list)}
                study.enqueue_trial(warm_start_params)

            except Exception as e:
                logging.warning(f"Warm start failed: {str(e)}")

        if approach in (['sequent', 'init_sequent']) and epoch > 0:
            try:
                # get best trial from previous study
                best_params = study.best_params
                warm_start_params = {"p_" + str(idx): value for idx, value in best_params.items()}

                study.enqueue_trial(warm_start_params)
            except Exception as e:
                logging.warning(f"Warm start failed: {str(e)}")

        # callbacks to early stop, modify output etc.

        callbacks = []

        if verbose:
            verbose_optuna = VerboseOptuna(verbose_level=10)
            callbacks = callbacks + [verbose_optuna]

        if ftol_abs > 0 and max_no_improvement > 0:
            early_stopping_cb = EarlyStopOptuna(ftol_abs, max_no_improvement)
            callbacks = callbacks + [early_stopping_cb]

        # optimize for current epoch

        try:
            study.optimize(
                optuna_objective,
                n_trials=iters_epoch,
                callbacks=callbacks
            )
        except Exception as e:
            logging.warning("Epoche did not terminate successfully")

        print('Epoche ' + str(epoch + 1) + ' finished')

        # update model with best parameters
        
        if study.best_trial is not None:
            best_params = study.best_params

            # convert parameter values back to array format
            param_values = [best_params.get("p_" + str(ind), param[0]) for ind, param in enumerate(params_list)]

            # update model parameters
            model_iter.conf.params.loc[params_subset.index, 'value'] = param_values
            print('Epoche combined scoring: ' + str(study.best_value))

        # clip parameters to bounds

        optim_mask = (model_iter.conf.params['if_active'] == 1) & (model_iter.conf.params['fixed'] == 0)
        upper_mask = (model_iter.conf.params['value'] > model_iter.conf.params['upper']) & optim_mask
        model_iter.conf.params.loc[upper_mask, 'value'] = model_iter.conf.params.loc[upper_mask, 'upper']
        lower_mask = (model_iter.conf.params['value'] < model_iter.conf.params['lower']) & optim_mask
        model_iter.conf.params.loc[lower_mask, 'value'] = model_iter.conf.params.loc[lower_mask, 'lower']

        # save model if required

        if save_epoch:
            _save_model(model_iter, model, ds, save_epoch)

    return model_iter


