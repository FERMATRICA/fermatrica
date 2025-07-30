"""
Wrappers for model outer layer optimization (local algos).

Both global and local algorithms are supported. As for now two derivative-free algos are included:
1. COBYLA constrained optimization by linear approximations (local)
2. PyGad genetic algorithm (global)

However, FERMATRICA architecture allows fast and simple adding new algorithms, and some
algos could be added later.

Derivative-free approach allows optimising without calculating (analytical) gradient what could be
very complex and time-consuming. However, some derivative algo (e.g. GS) could be added later
at least for some most popular transformations.
"""

import copy
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import nlopt
import nevergrad as ng
from smt.surrogate_models import RBF

from fermatrica.model.model import Model
from fermatrica.optim.objective import ObjectiveFunction, ObjectiveRBF, ObjectiveNG, VerboseNG, _save_model


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

    if params_subset.empty:
        logging.warning("No active non-fixed parameters to optimize")
        return model

    for i in range(epochs):

        if_error = False

        # rebuild objective function every epoch to update model.conf
        objective_fun = ObjectiveFunction(ds=ds
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


def optimize_local_bobyqa(ds: pd.DataFrame
                          , model: "Model"
                          , revert_score_sign: bool = False
                          , verbose: bool = True
                          , epochs: int = 3
                          , iters_epoch: int = 300
                          , error_score: float = 1e+12
                          , ftol_abs: float = .01):
    """
    Wrapper for BOBYQA constrained optimization by local approximations algorithm (local). BOBYQA
    is derivative-free, so no analytical gradient is required.

    BOBYQA minimizes score, so be careful to revert score if it is designed as maximizer before use
    or with argument `revert_score_sign=True`.

    Epochs are required to shuffle params a bit and to help algo get out from local optimum (sometimes)
    and to speed up calculation.

    Early stop threshold is defined as minimum absolute score gain per iteration. However, algo doesn't
    respect it directly, so don't be upset to see it working with much lesser gain per iteration.

    Local algorithms could be sensitive to starting (initial) values and to the number of optimising
    params. If params number is big enough (say, dozens) it is better to use it chunks, tuning
    one portion of params after another. Params for every chunk should be chosen wisely and not
    just first 10 or 20 params, them second 10 or 20 etc.

    Nlopt BOBYQA implementation is preferred before `scipy.optimize.minimize(method='BOBYQA')` because
    latter doesn't respect constraints accurately.

    However, Nlopt BOBYQA has its own annoying issue. If improvement is (much?) below machine precision,
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

    if params_subset.empty:
        logging.warning("No active non-fixed parameters to optimize")
        return model

    for i in range(epochs):

        if_error = False

        # rebuild objective function every epoch to update model.conf
        objective_fun = ObjectiveFunction(ds=ds
                                          , model=model
                                          , revert_score_sign=revert_score_sign
                                          , error_score=error_score
                                          , verbose=verbose)

        start_values = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]['value'].values.tolist()

        opt_obj = nlopt.opt(nlopt.LN_BOBYQA, len(start_values))

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


def optimize_local_sbplx(ds: pd.DataFrame
                         , model: "Model"
                         , revert_score_sign: bool = False
                         , verbose: bool = True
                         , epochs: int = 3
                         , iters_epoch: int = 300
                         , error_score: float = 1e+12
                         , ftol_abs: float = .01):
    """
    Wrapper for Sblpx / Subplex constrained optimization by local approximations algorithm (local). Sblpx / Subplex
    is derivative-free, so no analytical gradient is required.

    Sblpx / Subplex minimizes score, so be careful to revert score if it is designed as maximizer before use
    or with argument `revert_score_sign=True`.

    Epochs are required to shuffle params a bit and to help algo get out from local optimum (sometimes)
    and to speed up calculation.

    Early stop threshold is defined as minimum absolute score gain per iteration. However, algo doesn't
    respect it directly, so don't be upset to see it working with much lesser gain per iteration.

    Local algorithms could be sensitive to starting (initial) values and to the number of optimising
    params. If params number is big enough (say, dozens) it is better to use it chunks, tuning
    one portion of params after another. Params for every chunk should be chosen wisely and not
    just first 10 or 20 params, them second 10 or 20 etc.

    Nlopt Sblpx / Subplex implementation is preferred before `scipy.optimize.minimize(method='Sblpx / Subplex')` because
    latter doesn't respect constraints accurately.

    However, Nlopt Sblpx / Subplex has its own annoying issue. If improvement is (much?) below machine precision,
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

    if params_subset.empty:
        logging.warning("No active non-fixed parameters to optimize")
        return model

    for i in range(epochs):

        if_error = False

        # rebuild objective function every epoch to update model.conf
        objective_fun = ObjectiveFunction(ds=ds
                                          , model=model
                                          , revert_score_sign=revert_score_sign
                                          , error_score=error_score
                                          , verbose=verbose)

        start_values = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]['value'].values.tolist()

        opt_obj = nlopt.opt(nlopt.LN_SBPLX, len(start_values))

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


def optimize_local_rbf(ds: pd.DataFrame
                       , model: "Model"
                       , revert_score_sign: bool = False
                       , verbose: bool = True
                       , epochs: int = 3
                       , iters_epoch: int = 300
                       , error_score: float = 1e+12
                       , cores: int = 4
                       , ftol_abs: float = .01):
    """
    Surrogate-model-based optimization with parallel evaluation.
    Bayesian optimization using RBF surrogate model and Latin Hypercube Sampling.

    :param ds: dataset
    :param model: Model object
    :param revert_score_sign: if the score is the bigger, the better; this algo minimizes score
    :param verbose: print diagnostic or progress information
    :param epochs: number of epochs
    :param iters_epoch: number of objective function calculations per epoch
    :param error_score: extremely big value to be used as score if fit_predict returns None (error)
    :param cores: processor cores to be used in parallel computing
    :param ftol_abs: early stop threshold: minimum absolute score gain per iteration, see description
    :return: tuned Model

    """

    params_subset = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]
    start_values = params_subset['value'].values.tolist()

    if params_subset.empty:
        logging.warning("No active non-fixed parameters to optimize")
        return model

    if cores > (cpu_count() - 1):
        cores = (cpu_count() - 1)
        logging.warning('Cores number for parallel computing is set too high for this computer. ' +
                        'Reset to ' + str(cores))

    # extract bounds and parameter names

    n_params = len(params_subset)
    param_names = params_subset.index.tolist()

    lb = params_subset['lower'].values
    ub = params_subset['upper'].values

    # generate initial points around model's current parameters

    def generate_initial_points(n_points):

        std_dev = (ub - lb) * 0.05

        points = np.tile(start_values, (n_points, 1))
        points += np.random.normal(0, std_dev, points.shape)
        points = np.clip(points, lb, ub)

        points[0] = start_values

        return points

    # objective function wrapper

    model_iter = copy.deepcopy(model)
    model_iter.obj.transform_lhs_fn = None

    objective_fun = ObjectiveFunction(ds=ds,
                                      model=model_iter,
                                      revert_score_sign=revert_score_sign,
                                      error_score=error_score,
                                      verbose=False)

    # parallel evaluation function

    def evaluate_points(params):
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(objective_fun, params))
        return np.array(results)

    # initialize surrogate model

    n_start = min(2 * n_params, iters_epoch)
    param_train = generate_initial_points(n_start)
    score_train = evaluate_points(param_train)

    sm = RBF(d0=2 * n_params   # d0 controls basis function width
             , reg=1e-6
             , print_global=False
             , print_training=False
             , print_prediction=False)
    sm.set_training_values(param_train, score_train)
    sm.train()

    # main loop over epochs

    for epoch in range(epochs):
        if_error = False

        try:

            # batch acquisition via multi-start optimization
            batch_points = []

            optimize_surrogate = ObjectiveRBF(param_train=param_train
                                              , score_train=score_train
                                              , ub=ub
                                              , lb=lb
                                              , iters_epoch=iters_epoch
                                              , ftol_abs=ftol_abs
                                              , sm_object=sm)

            # parallel batch point generation

            with ProcessPoolExecutor(max_workers=cores) as executor:
                futures = [executor.submit(optimize_surrogate) for _ in range(cores * 10)]
                batch_points = [f.result() for f in futures]

            # evaluate batch in parallel

            new_score = evaluate_points(batch_points)

            # update training data

            param_train = np.vstack([param_train, batch_points])
            score_train = np.hstack([score_train, new_score])
            sm.set_training_values(param_train, score_train)
            sm.train()

            # update model with best parameters

            best_idx = np.argmin(score_train)
            best_params = param_train[best_idx]
            best_score = score_train[best_idx]
            model_iter.conf.params.loc[param_names, 'value'] = best_params

            # logging

            if verbose and (epoch + 1) % 1 == 0:
                score_print = (round(best_score, 5)
                               if abs(best_score) < 500
                               else round(best_score, 0))
                timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")

                print('Combined scoring : ' + str(score_print) + '; epoch : ' + str(
                    epoch + 1) + ' : ' + timestamp)

        except Exception as e:
            logging.warning(f"Epoche terminated unsuccessfully")
            if_error = True

        # boundary enforcement

        model.conf.params.loc[model.conf.params['value'] > model.conf.params['upper'], 'value'] =\
            model.conf.params[model.conf.params['value'] > model.conf.params['upper']]['upper']
        model.conf.params.loc[model.conf.params['value'] < model.conf.params['lower'], 'value'] =\
            model.conf.params[model.conf.params['value'] < model.conf.params['lower']]['lower']

        if if_error:
            break

    return model_iter


def optimize_local_ncma(ds: pd.DataFrame,
                        model: "Model",
                        revert_score_sign: bool = False,
                        verbose: bool = True,
                        epochs: int = 1,
                        iters_epoch: int = 1000,
                        error_score: float = 1e+12,
                        cores: int = 4,
                        approach: str = 'init_sequent',
                        save_epoch: bool | str = False):
    """
    Wrapper for CMA-ES optimization algorithm using nevergrad library. CMA-ES is a derivative-free
    global optimizer that maintains search history between iterations. It can handle bounds
    and is robust to noisy objective functions.

    Early stop threshold is implemented via callback mechanism.

    :param ds: dataset
    :param model: Model object
    :param revert_score_sign: if the score is the bigger, the better; this algo minimizes score
    :param verbose: print diagnostic or progress information
    :param epochs: number of epochs; each epoch is independent and returns its own score and tuned params
    :param iters_epoch: number of iterations per epoch
    :param error_score: extremely bad value to be used as score if fit_predict returns None (error)
    :param cores: processor cores to be used in parallel computing
    :param approach: 'void', 'init', 'sequent', 'init_sequent'
    :param save_epoch: if every epoch to be saved or path to the folder to save tuned models
    :return: tuned Model of the last epoch
    """

    params_subset = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]

    if params_subset.empty:
        logging.warning("No active non-fixed parameters to optimize")
        return model

    # set up multiprocessing

    if cores > (cpu_count() - 1):
        cores = cpu_count() - 1
        logging.warning('Cores number for parallel computing is set too high for this computer. '
                        f'Reset to {cores}')

    # get parameter bounds

    lwr = params_subset['lower'].values
    upr = params_subset['upper'].values

    lwr = np.nan_to_num(lwr, nan=-1e+8)
    upr = np.nan_to_num(upr, nan=1e+8)

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

    # get start values

    start_values = model.conf.params.loc[params_subset.index, 'value'].values
    param_names = params_subset.index.astype(str).tolist()

    best_score = objective_fun(params_cur=start_values.tolist())

    # main epoch loop

    with ProcessPoolExecutor(max_workers=cores) as executor:

        for epoch in range(epochs):

            # set up search space

            if approach == 'void' or (approach == 'sequent' and epoch == 0):
                start_values_cur = [np.random.uniform(bound[0], bound[1]) for bound in zip(lwr, upr)]
            else:
                start_values_cur = copy.deepcopy(start_values)

            search_space = ng.p.Instrumentation(**{
                name: ng.p.Scalar(lower=low, upper=high, init=val)
                for name, low, high, val in zip(param_names, lwr, upr, start_values_cur)
            })

            objective_ng = ObjectiveNG(param_names=param_names, objective_fun=objective_fun)

            # initialize CMA-ES

            optimizer = ng.optimizers.CMA(
                parametrization=search_space,
                budget=iters_epoch,
                num_workers=cores,
            )

            # callbacks

            if verbose:
                verbose_cb = VerboseNG(10)
                optimizer.register_callback('tell', verbose_cb)

            # run optimization

            try:
                rtrn = optimizer.minimize(objective_ng
                                          , executor=executor
                                          # , batch_mode=True
                                          , verbosity=0)
                best_params = rtrn.value

            except Exception as e:
                logging.warning("Epoche did not terminate successfully")
                best_params = start_values

            print('Epoche ' + str(epoch + 1) + ' finished')

            # update model with the best parameters

            if isinstance(best_params, tuple):
                params = []
                for par in param_names:
                    param_value = best_params[1][par]
                    params.append(param_value)

                score_cur = objective_fun(params_cur=params)

                if approach in ['init_sequent', 'sequent'] and score_cur < best_score:
                    start_values = params
                    best_score = score_cur
                elif approach in ['init_sequent', 'sequent'] and score_cur >= best_score:
                    params = start_values
                model_iter.conf.params.loc[params_subset.index, 'value'] = params
            else:
                logging.warning("Epoche did not terminate successfully")

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
