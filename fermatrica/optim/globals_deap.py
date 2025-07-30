"""
Wrappers for model outer layer optimization (global algos made with DEAP).

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
import ipyparallel.serialize.codeutil
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

import numpy as np
import pandas as pd
from deap import creator, base, tools
from scipy.stats import qmc

from fermatrica.model.model import Model
from fermatrica.basics.basics import fermatrica_error
from fermatrica.optim.objective import ObjectiveFunction, _save_model


def optimize_global_de(ds: pd.DataFrame,
                       model: "Model",
                       revert_score_sign: bool = False,
                       verbose: bool = True,
                       epochs: int = 1,
                       iters_epoch: int = 300,
                       pop_size: int = 50,
                       mutation: float | tuple | list = 0.8,
                       crossover: float = 0.9,
                       pop_init_method: str = 'random',
                       mut_method: str = 'random',
                       seed: int | None = None,
                       max_no_improvement: int = 20,
                       if_init: bool = False,
                       ftol_abs: float = 0,
                       error_score: float = 1e+12,
                       cores: int = 4,
                       save_epoch: bool | str = True) -> "Model":
    """
    Differential Evolution (DE) optimizer for global parameter optimization using the DEAP library.

    Features:
    - Parallel evaluation via multiprocessing
    - Adaptive mutation strategies (DE/rand/1, DE/best/1)
    - Population initialization with LHS/Sobol/Halton sampling
    - Early stopping for convergence detection
    - Support for warm-start optimization with initial values

    Parameters:
        ds (pd.DataFrame): Input dataset for model evaluation
        model (Model): Model object containing parameters to optimize
        revert_score_sign (bool): Invert objective function sign (for maximization). Default: False
        verbose (bool): Enable progress logging. Default: True
        epochs (int): Number of complete optimization restarts. Default: 1
        iters_epoch (int): Iterations per epoch. Default: 300
        pop_size (int): Population size. Default: 50
        mutation (float | tuple | list):
            - float: Fixed mutation factor F (e.g., 0.8)
            - tuple/list: Dynamic range [min, max] for F (e.g., (0.5, 1.0))
        crossover (float): Probability of crossover CR. Default: 0.9
        pop_init_method (str): Population initialization method:
            - 'random': Uniform random sampling
            - 'lhs': Latin Hypercube Sampling
            - 'sobol': Sobol sequence
            - 'halton': Halton sequence. Default: 'random'
        mut_method (str): Mutation strategy:
            - 'random': DE/rand/1 (classic)
            - 'best': DE/best/1 (uses the best individual). Default: 'random'
        seed (int | None): Random seed for reproducibility. Default: None
        max_no_improvement (int): Generations without improvement for early stopping. Default: 20
        if_init (bool): Use model's current parameters as initial population. Default: False
        ftol_abs (float): Absolute tolerance for improvement detection. Default: 0
        error_score (float): Value to return on evaluation errors. Default: 1e+12
        cores (int): CPU cores for parallel evaluation. Default: 4
        save_epoch (bool | str):
            - True: Save results to default location
            - str: Custom save path
            - False: Disable saving. Default: True

    Returns:
        Model: Optimized model with updated parameters

    Example:
        >>> best_model = optimize_global_de(
        ...     ds=data,
        ...     model=my_model,
        ...     pop_init_method='lhs',
        ...     mutation=(0.5, 1.0),
        ...     mut_method='combined',
        ...     epochs=5,
        ...     cores=8,
        ...     save_epoch='global_t'
        ... )
    """

    # problem setup

    params_subset = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]

    if params_subset.empty:
        logging.warning("No active non-fixed parameters to optimize")
        return model

    bounds = list(zip(
        np.nan_to_num(params_subset['lower'].values, nan=-1e8),
        np.nan_to_num(params_subset['upper'].values, nan=1e8)
    ))

    n_dim = len(bounds)

    if seed is None:
        rng = np.random.default_rng(seed)
        seed = rng.integers(0, int(1e6))

    if isinstance(mutation, tuple) or isinstance(mutation, list):

        if len(mutation) != 2:
            fermatrica_error('If mutation is a tuple or list it should contain two numbers')
        if mutation[1] <= mutation[0]:
            fermatrica_error('If mutation is a tuple or list the second number should be greater than the first one')
        if mutation[1] > 2:
            fermatrica_error('If mutation is a tuple or list the second number (upper bound) should be not greater than 2')
        if mutation[0] <= 0:
            fermatrica_error('If mutation is a tuple or list the first number (lower bound) should be greater than 0')
    else:
        if mutation > 2 or mutation <= 0:
            fermatrica_error('If mutation is a number it should be in (0, 2]')

    # DEAP framework initialization

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # run with multiprocessing

    with Pool(processes=cores) as pool:

        toolbox.register("map", pool.map)

        # genetic operators
        # not very clean approach with nonlocal variables in nested functions, but could be the only or simpler way

        def create_individual_random():
            """Generate 'individual' with random approach"""

            return [np.random.uniform(bound[0], bound[1]) for bound in bounds]

        def create_individual_lhs():
            """Generate 'individual' with LHS method"""

            sampler = qmc.LatinHypercube(d=n_dim)
            sample = sampler.random(n=1)

            return [bounds[i][0] + (bounds[i][1] - bounds[i][0]) * sample[0][i] for i in range(n_dim)]

        def create_individual_sobol():
            """Generate 'individual' with Sobol method"""

            sampler = qmc.Sobol(d=n_dim, scramble=True, seed=seed)
            sample = sampler.random(n=1)

            return [bounds[i][0] + (bounds[i][1] - bounds[i][0]) * sample[0][i] for i in range(n_dim)]

        def create_individual_halton():
            """Generate 'individual' with Halton method"""

            sampler = qmc.Halton(d=n_dim, scramble=True, seed=seed)
            sample = sampler.random(n=1)

            return [bounds[i][0] + (bounds[i][1] - bounds[i][0]) * sample[0][i] for i in range(n_dim)]

        # differential evolution operators

        def get_adaptive_params(gen: int) -> Tuple[float, float]:
            """Calculates mutation (F) and CR for current generation"""

            # mutation (F)
            if isinstance(mutation, tuple):
                mut_cur = mutation[1] - (mutation[1] - mutation[0]) * (gen / iters_epoch) ** 0.5
            else:
                mut_cur = mutation

            # CR (crossover)
            if isinstance(crossover, tuple):
                cr = crossover[0] + (crossover[1] - crossover[0]) * (gen / iters_epoch) ** 0.3
            else:
                cr = crossover

            return mut_cur, cr

        def mutate(individual: np.ndarray,
                   population: List[np.ndarray],
                   gen: int) -> Tuple[np.ndarray]:

            """
            Differential Evolution mutation operator.
            Creates a mutant vector by combining three randomly selected individuals.

            Parameters:
                individual (np.ndarray):  Target individual (not used in standard DE mutation,
                                          but required by DEAP interface)
                population (List[np.ndarray]): Current population of individuals
                mutation (float): Scaling factor (F) for differential mutation (typically in [0, 2])

            Returns:
                Tuple[np.ndarray]: Mutated individual wrapped in a tuple (DEAP requirement)

            Algorithm:
                1. Randomly select three distinct individuals from population (a ≠ b ≠ c)
                2. Compute mutant vector: mutant = a + F * (b - c)
                3. Apply element-wise clipping to maintain parameters within bounds
            """

            mut_cur, _ = get_adaptive_params(gen)

            if mut_method == 'best':
                mutant = tools.selBest(population, 1)[0]

            elif mut_method == 'combined':
                best = tools.selBest(population, 1)[0]
                a, b = tools.selRandom(population, 2)
                mutant = best + mut_cur * (a - b)

            else:  # random
                a, b, c = tools.selRandom(population, 3)
                mutant = a + mut_cur * (b - c)

            for i in range(n_dim):
                mutant[i] = np.clip(mutant[i], bounds[i][0], bounds[i][1])

            return mutant,

        def cx_binomial(ind1: np.ndarray,
                        ind2: np.ndarray,
                        cr: float) -> Tuple[np.ndarray]:
            """
            Binomial crossover operator for Differential Evolution.
            Performs gene-wise crossover between two individuals with probability CR.

            Parameters:
                ind1 (np.ndarray): First parent individual (modified in-place)
                ind2 (np.ndarray): Second parent individual (modified in-place)
                cr (float): Crossover probability in [0, 1] range

            Returns:
                Tuple[np.ndarray]: One offspring individuals

            Algorithm:
                1. For each gene position:
                   a. Generate random number in [0, 1)
                   b. If random < CR: swap genes between parents
                   c. Else: keep original genes
                2. Return modified individual

            Note:
                - Implements standard DE binomial crossover
                - Operates directly on numpy arrays
                - Maintains parameter bounds through vectorized operations
            """

            for i in range(n_dim):
                if np.random.rand() < cr:
                    ind1[i] = ind2[i]

            return ind1,

        # toolbox configuration

        toolbox.register("mutate", mutate)
        toolbox.register("mate", cx_binomial, cr=crossover)
        toolbox.register("select", tools.selBest)

        if pop_init_method == 'lhs':
            toolbox.register("individual", tools.initIterate, creator.Individual, create_individual_lhs)
        elif pop_init_method == 'sobol':
            toolbox.register("individual", tools.initIterate, creator.Individual, create_individual_sobol)
        elif pop_init_method == 'halton':
            toolbox.register("individual", tools.initIterate, creator.Individual, create_individual_halton)
        else:  # 'random'
            toolbox.register("individual", tools.initIterate, creator.Individual, create_individual_random)

        # optimization loop over epochs

        start_values = model.conf.params[(model.conf.params['if_active'] == 1) & (model.conf.params['fixed'] == 0)]['value'].values
        start_values = start_values.reshape(1, -1)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        for epoch in range(epochs):

            # recreate start conditions every epoch to make epochs independent

            stagnation_counter = 0
            best_values = None
            best_score = np.inf
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # recreate objective function every epoch, useful if every epoch to be independent

            model_iter = copy.deepcopy(model)
            model_iter.obj.transform_lhs_fn = None

            objective = ObjectiveFunction(
                ds=ds,
                model=model_iter,
                revert_score_sign=revert_score_sign,
                error_score=error_score,
                verbose=False
            )

            toolbox.register("evaluate", objective)

            #

            if if_init:
                initial_individuals = [creator.Individual(ind) for ind in start_values]

                remaining = pop_size - len(initial_individuals)

                if remaining > 0:
                    random_individuals = toolbox.population(n=remaining)
                    population = initial_individuals + random_individuals
                else:
                    population = initial_individuals[:pop_size]

            else:
                population = toolbox.population(n=pop_size)

            hof = tools.HallOfFame(1, similar=np.array_equal)
            stats = tools.Statistics(lambda ind: ind.fitness.values[0])

            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("std", np.std)

            # evaluate initial population

            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = (fit,)

            for gen in range(iters_epoch):

                # generate offspring

                offspring = []
                for idx in range(pop_size):

                    # mutation
                    mutant = toolbox.mutate(population[idx], population, gen)[0]

                    # crossover
                    trial = toolbox.clone(population[idx])
                    trial = toolbox.mate(trial, mutant)[0]

                    offspring.append(trial)

                # evaluate offspring

                fitnesses = toolbox.map(toolbox.evaluate, offspring)

                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = (fit,)

                # parent vs trial

                for idx in range(pop_size):
                    if offspring[idx].fitness.values[0] < population[idx].fitness.values[0]:
                        population[idx] = offspring[idx]

                # update HOF (Hall of Fame)

                hof.update(population)

                # update best solution

                current_best = hof[0].fitness.values[0]

                if current_best < (best_score - ftol_abs):

                    best_score = current_best
                    best_values = hof[0]
                    stagnation_counter = 0

                else:
                    stagnation_counter += 1

                # early stop

                if stagnation_counter >= max_no_improvement:
                    if verbose:
                        print("Early stop at step " +  str(gen + 1))
                    break

                # logging

                if verbose and (gen + 1) % 10 == 0:
                    score_print = (round(best_score, 5)
                                   if abs(best_score) < 500
                                   else round(best_score, 0))
                    timestamp = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")

                    print('Combined scoring : ' + str(score_print) + '; epoch : ' + str(epoch + 1) + '; step : ' + str(
                        gen + 1) + ' : ' + timestamp)

            print('Epoche ' + str(epoch + 1) + ' finished')

            if (isinstance(best_values, np.ndarray)) or (isinstance(best_values, list)):
                model_iter.conf.params.loc[
                    (model_iter.conf.params['if_active'] == 1) & (model_iter.conf.params['fixed'] == 0), 'value'] = best_values
            else:
                logging.warning("Epoche did not terminate successfully")

            optim_mask = (model_iter.conf.params['if_active'] == 1) & (model_iter.conf.params['fixed'] == 0)
            upper_mask = (model_iter.conf.params['value'] > model_iter.conf.params['upper']) & optim_mask
            model_iter.conf.params.loc[upper_mask, 'value'] = model_iter.conf.params.loc[upper_mask, 'upper']
            lower_mask = (model_iter.conf.params['value'] < model_iter.conf.params['lower']) & optim_mask
            model_iter.conf.params.loc[lower_mask, 'value'] = model_iter.conf.params.loc[lower_mask, 'lower']

            # save epoch results

            if save_epoch:
                _save_model(model_iter, model, ds, save_epoch)

    return model_iter
