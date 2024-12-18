# Marketing Mix Modeling with FERMATRICA

[_Russian version below_](#RU)

## Table of Contents

1. [Concept: Ideology of Marketing Mix Modeling](#part_1)

   - [Problem Statement, Agreement with Client on Key Points, Specifics, and Necessary Details](#start)
   - [Data Collection, ETL, Preprocessing](#data_collection)
   - [Feature Engineering and Exploratory Data Analysis (EDA)](#feature_engineering)
   - [Initial Model Building and Review of Results](#model_start)
   - [Model Refinement](#tuning)
     - [Manual Model Refinement](#hand_tuning)
     - [Model Refinement with Optimizer](#optimisation)
   - [Export and Presentation of Results in a Client-Friendly Format](#results)

2. [Step-by-Step Guide: How to Use FERMATRICA Framework for Modeling](#part_2)

3. [Useful Links](#part_3) 

---------------------------------------------------------------
---------------------------------------------------------------

## <a name="part_1"></a>Concept: Ideology of Marketing Mix Modeling

### <a name="start"></a>Problem Statement, Agreement with Client on Key Points, Specifics, and Necessary Details

Communication with the client's team and the client, setup meetings, briefing.

[To Practice: Step 1](#step_1)

### <a name="data_collection"></a>Data Collection, ETL, Preprocessing

- Think about what data could be useful
- Request data from the client
- Request data from colleagues in other departments
- Find data ourselves
- Load the data (ETL)
- Verify accuracy
- Merge into blocks and then into a single dataset (preprocessing)

Data can be loaded using a separate Python script or scripts. Recommended, but not mandatory structure could be:

```
-[root]
    -code_py
        -adhoc
            -__init__.py
            -data.py
            -preproc.py
        -scripts
            -data.py
            -preproc.py
            -exploratory.ipynb
```

It is important that all variables are finally brought to the same granularity for correct merging into a single dataset.

> _Additionally:_  
> <small>If the model is panel-based, it is essential to accurately establish the correspondence not only between periods and granularity but also between brands and sub-brands, regions, or other groups of objects. This is called harmonization.</small>

[To Practice: Step 2](#step_2)

### <a name="feature_engineering"></a>Feature Engineering and Exploratory Data Analysis (EDA)

If necessary:

- Create aggregated variables from existing ones
- Create dummy variables
- Create weighting coefficients
- Clean certain independent variables from the influence of others
- Identify the seasonal component (initially clean the target from important factors that might interfere with correct seasonality identification)

Build charts, correlation matrices, and other necessary statistics to dive into the dynamics of the studied indicators, identifying data features and potential errors (another verification stage) as deep as possible.

### <a name="model_start"></a>Initial Model Building and Review of Results

Now, when ETL and EDA is complete, we have a theoretical understanding of the future model's structure (hypotheses regarding the inclusion or exclusion of various variables, potential transformations) and practical understanding of the data.

Let's build and evaluate the initial model considered as baseline. Review the results obtained to understand the direction to move next.

The FERMATRICA model consists of two contours:

1. **Inner Contour**: A linear model, either OLS or the panel one. The linear model is very fast, is easily interpretable, and is well researched statistically. However, few real world dependencies are linear in nature.
2. **Outer Contour**: Parametrized nonlinear transformations of independent variables performed before sending them as features to the inner contour, and transformations of the target variable performed before and after the inner contour.

The two-contour scheme, generally following the standard of marketing mix modeling, extends it and allows modeling complex dependencies and phenomena.

[To Practice: Step 3](#step_3)

Reviewing results involves detailed examination of the model from a statistical perspective and its alignment with assumptions and the substance of the studied process. Key points to consider:

- Model summary (signs, significance of coefficients)
- Chart of observed and predicted values of the dependent variable, residuals chart
- Results of autocorrelation, stationarity, and normality tests of residuals
- Forecast quality (MAPE and unbiased forecast test)
- Level of multicollinearity by VIF
- Decomposition (dynamics chart and aggregated by periods, contributions of variables)
- Adequacy of transformations (chart of original and transformed variable, ensuring no transformations distort their intended meaning)
- Marginal ROI curves from media investments, ROMI curves (relation of curves, budget level beyond which values become non-zero and at which saturation occurs, adequacy of slope degree)

[To Practice - Step 4](#step_4)

### <a name="tuning"></a>Model Refinement

[To Practice - Step 5](#step_5)

#### <a name="hand_tuning"></a>Manual Model Refinement

Make necessary changes to variable transformations or the variables themselves (e.g., setting a feature differently), we can manually adjust the model settings or source data.

Such changes can be made to test a hypothesis, for example, to adjust optimizer settings (e.g., changing hyperparameter value boundaries or deciding whether to include a variable and its transformation).

#### <a name="optimisation"></a>Model Refinement with Optimizer

Transformation parameters can be selected not only manually but also using numerical optimization - maximizing the score function reflecting model quality (considering model features and requirements).

### <a name="results"></a>Export and Presentation of Results in a Client-Friendly Format

Depending on the client's preferred format, export detailed model information to Excel or a presentation.

Additionally, a dashboard can be developed for client purposes.

[To Practice - Step 6](#step_6)

------------------------------------------------------------------------
------------------------------------------------------------------------

## <a name="part_2"></a>Step-by-Step Guide: How to Use the FERMATRICA Framework for Modeling

### <a name="step_1"></a>Step 1

Create a project folder with the following structure:

- ***data*** - Folder to store raw data for the project
- ***code_py*** - Folder to store preprocessed data, scripts, and modeling results:
  - **adhoc** - Folder for functions written specifically for this project. Can remain empty.
  - **scripts** - Main working folder where scripts for working on the model will live
  - <a name="data"></a>**data** - Folder with subfolder **data_processed**, to store preprocessed data ready for modeling
  - **model_data** - Folder to store:
    - Model configuration files (e.g., [*model_conf.xlsx*](#config))
    - Folders with saved models (possibly multiple levels). The model folder (itself a model object) will be created automatically after saving the model manually or after optimizer work. Folder name model_YYY-MM-DD_HHmmss consists of three files:
      - *dt_p.pkl.zstd* - Data file
      - *model_conf.xlsx* - Model configuration file
      - *model_objects.pkl.lzma* - Model objects and model evaluation results file
  - **results** - Folder to export model evaluation results in a user-friendly format
  - Additional files if required (e.g., dictionaries)
- ***reports*** - Folder for final results (presentations, calculators, etc., for demonstration and/or sending to the client)
- ***.idea*** - Folder indicating the conversion of the folder to a Python project (created automatically upon first opening the folder as a project in PyCharm)

[Back to Theory - Problem Statement, Agreement with Client on Key Points, Specifics, and Necessary Details](#start)


### <a name="step_2"></a>Step 2

Start filling folders:

- Place raw data in the [data](#data) folder
- In the code_py.scripts folder or the code folder, create a script to load and verify data. Load and verify data.
- Combine them into a single <a name="dataset"></a>dataset, saved in code_py.data.data_processed

[Back to Theory - Data Collection and Loading](#data_collection)


### <a name="step_3"></a>Step 3

Proceed to model building, starting with initial modeling.

Model assembly can be done in two ways: in Jupyter Notebook (Block **Load data and current model definition**) or using the script model.py.

Linear models, both with and without transformations, can be evaluated using FERMATRICA.

To evaluate the model, a prepared [dataset](#dataset) is required, and an Excel file with model configuration needs to be filled out. Create an Excel file [model_config.xlsx](#config) for this purpose.

<details>

**<summary>Filling the Model Configuration File - model_conf.xlsx (<a name="config"></a>click to expand for details)</summary>**

The Excel file has the following tabs:

- ***setup*** - General model settings:
  - **Y_var** - Name of the dependent variable
  - **model_type** - Model type (OLS, LME, LMEA, FE, default is OLS)
  - **period_var** - Variable defining the period
  - **price_var** - Price variable, default is price_distr (if prices are unknown, create a variable price_distr with a value of 1, specifying its name here)
  - **bs_key** - Name of the object group variable, usually the superbrand (if not a panel and the brand is single, create a variable superbrand in the dataset with a single value - the name of the modeled brand)

  For panel models and optionally:
  - **summary_type** - For budget summary options (available values sum - sum for the period and fin - end-of-period value, default is sum)
  - **units** - Units of measurement of the dependent variable, default is units
  - **conversion_var** - Conversion variable (e.g., if the dependent variable is not sales, but conclusions are to be made for sales)
  - **conversion_fun** - Conversion function (name of custom function to transition from the dependent variable to the variable of interest, e.g., sales)
  - <a name="cs"></a>**custom_scoring** - Manually specified objective function to replace the standard weighted sum of scores
  - **target_audience** - Target audience for TV, e.g., f25-34 (if several, specify on separate lines, then read into a list)
  - **target_superbrand** - Name of the main study brand (if not panelized and the brand is single, create a variable superbrand in the dataset with a single value - the name of the modeled brand)
  - **lme_method** - Optimization method in statsmodels.formula.api.mixedlm (lbfgs and powell; powell is more accurate for complex models, does not crash, works relatively slowly; lbfgs is relatively fast, preferred for relatively standard but large panel models, may crash for complex ones)
  - **fixed_effect_var** - Variable for entity existence (usually bs_key)
  - **exclude_from_curves** - Variables to exclude from dashboard rendering
  - **exclude_curve** - Variables to exclude from dashboard rendering

- ***LHS*** - Left-hand side transformations (usually needed for complex models; generally not required for simple models)

- ***params*** - Transformations with the variable being transformed, transformation function, parameters, their boundaries, starting values, inclusion flag, and additional settings:
  - **variable** - Name of the variable to be transformed
  - **fun** - Name of the transformation function (basic ones in `fermatrica.model.transform`, custom adhoc functions can be defined, specify the path, e.g., code_py.adhoc.model.adhoc_tv)
  - **arg** - Arguments of the transformation function - parameters for evaluation
  - **lower**, **upper**, **value** - Parameter values: lower and upper (required for optimization), current (also starting for optimization)
  - **type** - Variable type of the parameter (e.g., float64, int, bool, or str)
  - **fixed** - Optimization flag for the parameter (1 - yes, 0 - no, current value is used)
  - **if_active** - Flag for using the transformation and parameter (1 - yes, 0 - no, thus parameter not instantiated in the model object)
  - **index_vars** - Grouping variable (default is no grouping)
  - **index_aggr** - Grouping function (defined in `fermatrica.model.transform`, possible are: sum_kpi_coef, sum_kpi_coef_master, max, mfreq, default is no grouping or sum if index_vars is not empty)
  - **index_free_var** - Variable for splitting to apply transformation 
  - **set_start** - Use the formula for calculating the current parameter value (if formula exists)
  - **formula** - Formula for calculating the current parameter value (e.g., based on variables in the dataset ds)

- ***RHS*** - Variables potentially included in the model, with inclusion flags, expected signs, and weights reflecting their "importance" in the optimizer's work:
  - **token** - Name of the variable to be included in the linear model
    - If the variable is to be transformed, append the transformation function name separated by "_"
    - The variable in the linear model might represent a combination, record as a formula like I(), e.g., I(X1 + X2 + X3), where X1, X2, X3 are variables in the dataset
  - **fixed** - Placeholder for future, currently all 1 (for further use when functionality for variable set optimization inclusion evolves)
  - **if_active** - Inclusion flag for the variable in the model (1 - yes, 0 - otherwise)
  - **sign** - Preferred variable sign (1 - positive, -1 - negative)
  - **sign_weight** - Weight reflecting the relative importance of the variable's sign
  - **signif_weight** - Weight reflecting the relative importance of the variable's significance
  - **display_var** - How the variable should be named in analysis and reporting, e.g., dashboard or creating presentations (default duplicates token; aggregated if multiple with the same names)

- ***scoring*** - What is included in the objective function, optimized score, the metrics, and their weights:
  - **rmse** - Root Mean Squared Error - Smaller is better - if_invert = 1
  - **r_squared** - Coefficient of Determination (R-Squared) - Larger is better - if_invert = 0
  - **r_squared_adjusted** - Adjusted Coefficient of Determination - Larger is better - if_invert = 0
  - **t_value** - T-Statistics of regression coefficients considering weights - Larger is better - if_invert = 0
  - **p_value** - P-Values of regression coefficients considering weights - Smaller is better - if_invert = 1
  - **sign_simple** - Expected sign compliance considering weights - Smaller is better - if_invert = 1
  - **sign_t_value** - T-Statistics of sign compliance considering weights - Smaller is better - if_invert = 1
  - **vif** - Mean Variance Inflation Factor - Smaller is better - if_invert = 1
  - **durbin_watson_score** - Durbin-Watson autocorrelation test minus 2 - Smaller is better - if_invert = 1

> _Additionally:_
> - <small>If custom scoring in [**custom_scoring**](#cs) is set, score components should be set on this sheet and be active.</small>
> - <small>If wanting to invert full score (invert), set this in the optimization function call.</small>

- ***trans_path_df*** - This sheet will be auto-created after the optimizer's first run or can be preset.

    For each variable assign:
    - **variable** - Original variable from the dataset ds used in the model,
    - **variable_fin** - Transformed variable used in the model
    - **price** - Price per media impact unit for correct curve rendering (if non-media, leave the price cell empty),
    - **display_var** - How the variable is named in the dashboard or presentation creation.

</details>
<br/>

Each transformation specified on the `params` tab creates a new variable in the dataset, named by combining the original variable and transformation function names with an underscore. For example, the transformation `softmax` on variable `tv` creates the variable `tv_softmax`.

Transformations occur sequentially.

Transformation coefficients for the left-hand side are also specified on this tab, distinguished by an empty `variable` field and the value `coef` in the `fun` field. Coefficients apply in left-hand side transformation order.

<details>

**<summary>Main Functions for Variable Transformations (click to expand for details)</summary>**

Functions are specified on the `params` tab of the `model_conf.xlsx`.

##### Temporal Transformations

***lag()***: Variable lagging (time shift), delay

- **n** - Number of lags (by how many periods shift)
- **cval** - Value to fill initial gaps caused by lag. Default is 0

***mar()***: Right smoothing by moving average

- **n** - Number of lags (by how many periods shift)
- **if_scale** - Whether to weight the smoothed variable by average value in train + test

***adstock(), adstockp()***: Geometric adstock (delay, modeling delayed effect). The first variant is weighted, the second (`p`) is pure and resistant to new observations

- **a** - Decay rate (portion of impact in period t affecting t+1)

***adstockpd()***: Linear combination of two adstocks with different decay rates.

- **a1** - Decay rate of the first adstock
- **a2** - Decay rate of the second adstock
- **w2** - Coefficient for the second adstock in the linear combination (0 to 1). The first adstock is weighted by `1 - w2`

***dwbl(), dwblp()***: Decay / temporal transformation based on Weibull distribution (the second variant is pure, resistant to new observations)

- **dur** - Duration of the effect in periods (rounded to integer)
- **shape** - Parameter defining the curve shape (shape < 1 — C-shaped, shape > 1 — S-shaped)
- **scale** - Scale coefficient
- **norm** - Normalization flag, default is False

##### Saturation

***expm1(), expm1scaled()***: Exponential transformation with subtracting 1 from the result and weighing with median and standard deviation (second variant).

- **product** - Multiplier for the original variable before applying the exponent, making the curve smoother or flatter.

***softmax(), softmaxfull()***: Variants of logistic transformation (sigmoid), the first variant subtracts zero on the x-axis (so that with 0 on X, the Y value is also 0).

- **lambda** - Steepness parameter of the transformation; the smaller it is, the steeper the transformation.
- **avg** - Inflection point (the growth rate starts to slow down).
- **std** - Standard deviation of the original variable, used to clear lambda from the dimension of the original variable.

***logistic()***: Variant of logistic transformation (sigmoid).

- **steep** - Steepness parameter of the transformation.
- **ec50** - Parameter for reaching half saturation.
- **level** - Saturation level (used to clear other parameters from the dimension of the original variable).

***adbudg()***: Depending on the argument values, approximates either a power or sigmoid form.

- **steep** - Steepness parameter of the transformation.
- **ec50** - Parameter for reaching half saturation.
- **level** - Saturation level (used to clear other parameters from the dimension of the original variable).

...Plus a few other rarely used functions.

##### Complex Transformations

***arl(), arlp()***: Transformation combining saturation, adstock, and lag. In the first case, saturation is set by function `adbudg()`, in the second - by `logistic()`.

- **a** - Decay rate.
- **steep** - Steepness parameter of the transformation.
- **ec50** - Parameter for reaching half saturation.
- **level** - Saturation level (used to clear other parameters from the dimension of the original variable).
- **n_lag** - Number of lags (by how many periods to shift).
- **cval** - Value to fill the edges of the shifted vector if there is a lag.

! Note: In `arl()` and `arlp()` functions, saturation comes first, then adstock, and then the lag.

***price()***: Complex transformation calculating prices relative to the average in the group at the moment.

</details>
<br/>

Next, we load and evaluate the initial model without using the optimizer. This is done in three steps.

1. **Loading the Model.** Create a `fermatrica.Model` object by reading `model_conf.xlsx` during object initialization. For more complex model structures, adhoc code and the function for building a categorical model are passed as arguments at this stage.
    ```python
    import os
    import fermatrica as fm
   
    pth = os.path.join("code_py", "data", "model_conf.xlsx")
    model = fm.Model(path=pth,
                     custom_predict_fn=None,
                     adhoc_code=[code_py.adhoc.model],
                     ds=dt_p)
    ```

2. **Executing Outer Contour Transformations.** Model building and getting a forecast are done in two steps using the same pair of functions. This is necessary to improve performance for complex models. First, execute the outer contour transformations. The `set_start` argument initializes calculation of initial parameter values (where specified), handle it carefully.
    ```python
    model = fm.transform(ds=dt_p,
                         model=model,
                         set_start=True,
                         if_by_ref=True)
    ```

3. **Fitting Inner Contour and Getting a Forecast.** Once the transformations are done, execute the inner contour. At this stage, the internal linear model is built, and the transformations of the left part are performed both ways. You can run `fermatrica.fit_predict()` twice with different arguments - first to build the model and get a forecast on the train set, then to get a forecast on the entire dataset. The function `fermatrica.predict_ext()` returns the forecast as a dataframe with additional columns convenient for analysis.
    ```python
    pred, pred_raw, model = fm.fit_predict(dt_p, model, if_fit=True, return_full=True)

    dt_pred = fm.predict_ext(model, dt_p)
    ```

> _Additionally:_  <small>Example available soon: [examples will be ready soon]</small>

[Back to Theory - Initial Model Assembly and Review of Results](#model_start)

### <a name="step_4"></a> Step 4

Review the results and save if satisfactory.

To save the model, use `fermatrica.model.save` method (specify the path where results will be saved).

Results can be reviewed:

- In Jupyter Notebook (Blocks **Get results**, **Statistic and metrics**, **Observed and predicted**, **Decomposition and Waterfall**) if the model is built there.
- In the same script model.py using FERMATRICA functions for calculating required statistics and graphs (tests if needed, observed and predicted values graph of the dependent variable, residuals graph, forecast quality, decomposition). The model can be evaluated by re-running the script with new config settings.
- In the FERMATRICA_DASH dashboard if the model is saved.

Each approach has its pros and cons; the researcher can choose the preferred method.

[Back to Theory - Initial Model Assembly and Review of Results](#model_start)

### <a name="step_5"></a> Step 5

Work on the model.

#### Manually Change Parameters and Variable Composition

- Modify the variable list on the ***RHS*** sheet in the config (add new ones or "enable"/"disable" existing ones).
- Add/remove/change transformations on the ***params*** sheet and in the variable names included in the model on the ***RHS*** sheet in the config.
- Change transformation parameter values in the **value** column on the ***params*** sheet in the config.

Reevaluate the model and analyze the results.

This can be done in a separate .py script or within the model refinement in Jupyter Notebook or the model.py script.

> _Additionally:_  <small>Multiple transformations can be applied sequentially to one variable. For this, specify the order on the params sheet: transformation *a* is applied to the already transformed variable X with *b*, so on the params sheet first specify *b* for X, then *a* for X_b, and in the model it will be X_b_a, which should be specified in **token**.</small>

[Back to Theory - Model Refinement](#tuning)

#### Using Optimizer for Parameter Tuning

Transformation parameters can be selected using numerical optimization.

By default, the objective function (score) is formed as a linear combination of metrics specified in the config on the ***scoring*** sheet in the **metrics** column and the **if_active** column (needed ones should be "enabled"), with weights **weight**.
Alternatively, the objective function can be manually specified in the same file in [**custom_scoring**](#cs).

Local and global algorithms are supported; currently, these are the local COBYLA algorithm, called by `fermatrica.optimize_local_cobyla()`, and the genetic algorithm PyGAD, called by `fermatrica.optimize_global_ga()`.

<details>

**<summary>Optimizer Settings (click to expand for details)</summary>**

For the COBYLA local algorithm (***local***):
- *revert_score_sign* - Change the sign of the final score? Do this if the score is being maximized because COBYLA minimizes.
- *verbose* - Whether to print diagnostic information.
- *epochs* - Number of epochs - how many times the optimizer will run. In local optimizer, the next epoch inherits the previous one, thus the number of optimal models doesn't depend on the number of epochs and is always 1. The purpose of multiple epochs is "shuffling" the algorithm to better avoid local minima.
- *iters_epoch* - Maximum number of optimizer iterations within one epoch.
- *error_score* - Score in case of error. Set to a large number in case of a minimizing algorithm - with a positive sign.
- *ftol_abs* - Growth threshold of the optimization function after which the algorithm stops. COBYLA treats this rather willfully, continuing if it sees potential optimization (slope) but doesn't entirely ignore it.

For the PyGAD genetic algorithm (***global***):
- *revert_score_sign* - Change the sign of the final score? Do this if the score is being minimized because PyGAD maximizes.
- *verbose* - Whether to print diagnostic information.
- *epochs* - Number of epochs - how many times the optimizer will run, determines the number of evaluated models. In the global optimizer, epochs are independent, and the optimal variant is saved at the end of each.
- *iters_epoch* - Maximum number of optimizer iterations within one run.
- *pop_size* - Initial population size for the genetic algorithm (initial number of subject-models at algorithm start).
- *pmutation* - Probability of mutation (random changes in subject-models) in the genetic algorithm.
- *max_no_improvement* - Maximum number of iterations without recorded score improvement.
- *error_score* - Score in case of error. Set to a large number in case of a maximizing algorithm - with a negative sign.
- *cores* - Number of logical computer cores available for calculations.
- *save_epoch* - Path to the folder to save results after each epoch.

</details>
<br/>

> _Additionally:_  <small>Example available soon: [coming soon]</small>

Repeat Step 4 and Step 5 until achieving fully satisfactory results.

### <a name="step_6"></a> Step 6

Results export for the client can be:

- In a PowerPoint presentation
- In a dashboard

> _Additionally:_  
> - <small>The dashboard exists as a separate repository FERMATRICA_DASH</small>
> - <small>The slide generator is part of the FERMATRICA_REP repository</small>

[Back to Theory - Export and Presentation of Results in a Client-Friendly Format](#results)

Prepare a report on the model in the output folder.

------------------------------------------------------------------------
------------------------------------------------------------------------

## <a name="part_3"></a>Useful Links

- [Documentation](https://fermatrica.github.io/docs)
- [Source Code](https://github.com/FERMATRICA)
- [Samples](https://github.com/FERMATRICA/fermatrica/tree/master/samples)



------------------------------------
<a name="RU"></a>

# Моделирование маркетингового микса (MMM) с FERMATRICA

## Содержание

1.  [Идеология моделирования маркетингового микса](#ru_part_1)

    -   [Постановка задачи, согласование с клиентом основных моментов, специфики и необходимых деталей](#ru_start)

    -   [Сбор и подгрузка данных](#ru_data_collection) 
    
    -   [Генерация дополнительных переменных и предварительный анализ данных](#ru_feature_engineering)
    
    -   [Первичная сборка модели и просмотр результатов](#ru_model_start)
    
    -   [Доработка модели](#ru_tuning)
    
        -   [Ручная доработка модели](#ru_hand_tuning)
        
        -   [Доработка модели с использованием оптимизатора](#ru_optimisation)
        
    -   [Выгрузка и представление результатов в удобном клиенту формате](#ru_results)

2.  [Как пользоваться фреймворком FERMATRICA для моделирования - Пошаговая инструкция](#ru_part_2)

3. [Полезные ссылки](#ru_part_3) 



---------------------------------------------------------------
---------------------------------------------------------------

## <a name="ru_part_1"></a>Идеология моделирования маркетингового микса

### <a name="ru_start"></a>Постановка задачи, согласование с клиентом основных моментов, специфики и необходимых деталей

Переписка с клиентской командой и с клиентом, установочные встречи, брифинг.

[К практике - Шаг 1](#ru_step_1)

### <a name="ru_data_collection"></a>Сбор и подгрузка данных

-   подумать, что нужно
-   запросить у клиента
-   запросить у коллег из других подразделений
-   найти самим
-   подгрузить
-   проверить корректность
-   склеить в блоки и далее в единый датасет

Данные могут подгружаться в отдельном скрипте в Python.

Важно, чтобы предварительно все переменные были приведены к единой гранулярности для корректной склейки в единый датасет.

> _Дополнительно:_  
> <small>Если модель панельная, то нужно аккуратно установить соответствие не только между периодами и грануяцией, но и между брендами и суббрендами, регионами или другими группами объектов. Это называется гармонизация</small>

[К практике - Шаг 2](#ru_step_2)

### <a name="ru_feature_engineering"></a>Генерация дополнительных переменных (feature engineering) и предварительный анализ данных

При необходимости

-   создать агрегированные переменные из имеющихся,
-   создать фиктивные (dummy) переменные,
-   создать взвешивающие коэффициенты, веса,
-   очистить одни независимые переменные от влияния других,
-   выделить сезонную составляющую (предварительно очистив целевую от важных факторов, которые могут помешать корректному выделению сезонности)

Построить графики, матрицы парных корреляций и другие необходимые статистики для более полного понимания динамики исследуемых показателей, выявления особенностей в данных и потенциальных ошибок (ещё один этап проверки).

### <a name="ru_model_start"></a>Первичная сборка модели и просмотр результатов

Имея теоретическое представление о структуре будущей модели (гипотезы относительно включения и невключения различных переменных, потенциальных их преобразований) и изучив предварительно ряды данных, оцениваем первичную модель, которую будем считать базовой. Изучаем полученные результаты, чтобы понять, куда двигаться дальше.

Модель FERMATRICA состоит из двух контуров:

1. Внутренний контур - линейная модель, простая или панельная. Линейная модель быстро считается, легко интерпретируется и хорошо исследована статистически. Однако мало какие зависимости в жизни носят линейный характер.

2. Внешний контур - параметризованные нелинейные преобразования независимых переменных, совершаемые до отправки их в качестве фич во внутренний контур, а также преобразования целевой переменной, совершаемые до и после выполнения внутреннего контура.

Двухконтурная схема, в общем следуя стандарту моделирования маркетингового микса, расширяет его и позволяет моделировать сложные зависимости и явления.

[К практике - Шаг 3](#ru_step_3)

Просмотр результатов предполагает детальное изучение модели как с точки зрения статистики и соответствия предпосылкам, так и с точки зрения содержания, смысла исследуемого процесса. Нужно обратить внимание на следующие моменты:

-   саммари по модели (знаки, значимость коэффициентов)
-   график наблюдаемых и прогнозных значений зависимой переменной, график остатков
-   результаты тестов на автокорреляцию, стационарность и нормальность остатков
-   качество прогноза (MAPE и тест на несмещенность прогноза)
-   уровень мультиколлинеарности по VIF
-   декомпозиция (на графике динамики и агрегированная по периодам, вклады переменных)
-   адекватность преобразований (график исхдной и преобразованной переменной, нужно отсмотреть все, чтобы никакие преобразования не искажали смысла, который в них заложен)
-   кривые предельной отдами от медиаинвестиций, кривые ROMI (соотношение кривых, уровень бюджета, на котором значения становятся отличными от нуля и на котором происходит насыщение, адекватность степени пологости)

[К практике - Шаг 4](#ru_step_4)

### <a name="ru_tuning"></a>Доработка модели

[К практике - Шаг 5](#ru_step_5)

#### <a name="ru_hand_tuning"></a>Ручная доработка модели

Если есть необходимость, можно внести изменения нужно преобразования переменных или в сами переменные (например, иначе задать некоторый признак), мы можем внести ручные правки в модель (в настройки модели или в исходные данные).

Такие изменения можно вносить и для того, чтобы проверить какую-то гипотезу, например для дальнейшего изменения настроек оптимизатора (например, изменения границ значений гиперпараметров или целесообразности включения переменной, приобразования для переменной).

#### <a name="ru_optimisation"></a>Доработка модели с использованием оптимизатора

Параметры преобразований могет быть подобраны не только вручную, но и с помощью численной оптимизации - максимизируется скор - функция, которая отражает качество модели (подбирается с учетом особенностей модели и требований к ней)

### <a name="ru_results"></a>Выгрузка и представление результатов в удобном клиенту формате

В зависимости от формата, предпочитаемого клиентом, могут быть выгружены детали по модели в excel или в презентацию.

Также может быть отдельно разработан дашборд для целей клиента.

[К практике - Шаг 6](#ru_step_6)

------------------------------------------------------------------------
------------------------------------------------------------------------

## <a name="ru_part_2"></a>Как пользоваться фреймворком FERMATRICA для моделирования - Пошаговая инструкция

### <a name="ru_step_1"></a> ШАГ 1

Создаем проект - папку следующей структуры:

-   ***data*** - папка, в которой будут хранится исходные данные для проекта
-   ***code_py*** - папка, в которой будут хранится предобработанные данные, скрипты и результаты моделировани:
    -   **adhoc** - папка, в которой будут жить функции, написанные специально для этого проекта. Может быть пустой
    -   **scripts** - основная рабочая папка, в ней будут жить скрипты для работы над моделью
    -   <a name="ru_data"></a>**data** - папка с подпапкой **data_processed**, там будут жить предобработанные данные, готовые для работы над моделью
    -   **model_data** - папка, в которой будут лежать
        -   файлы с настройками модели (например, [*model_conf.xlsx*](#ru_config))
        -   папки с сохраненными моделями (могут быть в несколько уровней). Папка с моделью (она и есть объект модели) будет создана автоматически после сохранения модели вручную или после работы оптимизатора. Название папки model_YYY-MM-DD_HHmmss состоит из трех файлов:
            -   *dt_p.pkl.zstd* - файл с данными
            -   *model_conf.xlsx* - файл с настройками модели
            -   *model_objects.pkl.lzma* - файл с объектами модели и результатами оценки модели
    -   **results** - папка, в которую можно выгрузить результаты оценки модели в удобном для дальнейшего использования виде
    -   дополнительные файлы, если требуются (например, словари)
-   ***reports*** - папка, в которой будут жить финальные результаты (презентации, калькуляторы и тд для демонстрации и/или отправки заказчику)
-   ***.idea*** - папка, которая отражает, что просто папка стала Python-проектом (создается автоматически при первом открытии папки как проекта в PyCharm)

[Назад к теории - Постановка задачи, согласование с клиентом основных моментов, специфики и необходимых деталей](#ru_start)

### <a name="ru_step_2"></a> ШАГ 2

Начинаем заполнять папки:

-   Исходные данные кладем в папку [data](#ru_data)
-   В папке code_py.scripts или в папке code зоздаем скрипт для подгрузки и проверки данных. Подгружаем и проверяем данные.
-   собираем их в единый <a name="ru_dataset"></a>датасет, который сохраняем в code_py.data.data_processed

[Назад к теории - Сбор и подгрузка данных](#ru_data_collection)

### <a name="ru_step_3"></a>ШАГ 3

Переходим к построению модели. Начинаем с первичного моделирования.

Сборка модели может быть проведена по двум схемам: в Jupyter Notebook (Блок **Load data and current model definition**) или с помощью скрипта model.py.

Линейная модель как без преобразований, так и с ними может быть оценена с помощью FERMATRICA. 

Для оценки модели требуется подготовленный [датасет](#ru_dataset), а также нужно заполнить Excel-файл с настройками (конфигурацией модели). Для этого создаем Excel-файл [model_config.xlsx](#ru_config)

<details>

**<summary>Заполнение файла с <a name="ru_config"></a> настройками модели - model_conf.xlsx (чтобы узнать подробности, нажмите на треугольник)</summary>**


Excel-файл, который содержит следующие листы:

-   ***setup*** - общие настройки модели:

    -   **Y_var** - имя зависимой переменной
    -   **model_type** - тип модели (OLS, LME, LMEA, FE, по умолчанию OLS)
    -   **period_var** - переманная, задающая период
    -   **price_var** - переменная цены, по умолчанию price_distr (если цены неизвестны, необходимо создать переменную price_distr со значением 1, название которой указать здесь)
    -   **bs_key** - название переменной групп объёктов, как правило, бренда - superbrand (если не панель и бренд один, необходимо создать переменную superbrand в датасете с единым значением - названием моделируемого бренда)

    Для панельных моделей и опционально:

    -   **summary_type** - для сводных данных по бюджетным опциям (доступные значения sum - сумма за период и fin - значение на конец периода, по умолчанию sum)
    -   **units** - единицы измерения зависимой переменной, по умолчанию units
    -   **conversion_var** - переменная конверсии (например, если зависимая переменная не продажи, а выводы хотим делать для продаж)
    -   **conversion_fun** - функция конверсии (имя кастомной функции, по которой булет произведен переход от зависимой переменной к интересующей, например, продажам)
    -   <a name="ru_cs"></a>**custom_scoring** - прописанная вручную целевая функция, которая будет использована вместо стандартной взвеженной суммы скоров
    -   **target_audience** - целевая аудитория для ТВ, например, f25-34 (если несколько, то прописываются в несколько строк, тогда далее считываюия в list)
    -   **target_superbrand** - название основного для исследования бренда (если не панель и бренд один, необходимо создать переменную superbrand в датасете с единым значением - названием моделируемого бренда)
    -   **lme_method** - метод оптимизации в statsmodels.formula.api.mixedlm (lbfgs и powell; powell более точен в сложных моделях, не падает, работает относительно медленно; lbfgs - работает относительно быстро, предпочтителен для относительно стандартных, но больших панельных моделей, для сложных может падать)
    -   **fixed_effect_var** - какая переменная отвечает за наличие сущностей (обычно bs_key)
    -   **exclude_from_curves** - переменные, которые хотим исключить из отрисовки в дашборде
    -   **exclude_curve** - переменные, которые хотим исключить из отрисовки в дашборде

-   ***LHS*** - преобразования левой части (обычно нужно для сложных моделей; для простых моделей, как правило, не требуется)

-   ***params*** - преобразования с указанием трансформируемой переменной, функции преобразования, параметров, их границ и стартовых значений, флага включения и других дополнительных настроек:

    -   **variable** - имя переменной, которая будет преобразована
    -   **fun** - имя функции преобразования (базовые заданы в `fermatrica.model.transform`, можно задать свои adhoc функции, тогда имя нужно прописать с путем, например, code_py.adhoc.model.adhoc_tv)
    -   **arg** - аргументы функции преобразования - параметры для оценивания
    -   **lower**, **upper**, **value** - значения параметров: нижнее и верхнее (нужно при оптимизации), текущее (оно же стартовое при оптимизации)
    -   **type** - типа переменной параметра (например, float64, int, bool или str)
    -   **fixed** - флаг на необходимость оптимизировать параметр (1 - если да, 0 - если нет - тогда будет взято текущее значение)
    -   **if_active** - флаг на использование преобразования и параметра (1 - если да, 0 - если нет, тогда в объекте модели не будет этих параметров)
    -   **index_vars** - переменная группировки (по умолчанию группировки не будет)
    -   **index_aggr** - функция группировки (заданы в `fermatrica.model.transform`, возможны следующие: sum_kpi_coef, sum_kpi_coef_master, max, mfreq, по умолчанию группировки нет или sum, если в index_vars не пусто)
    -   **index_free_var** - переменная, по которой будет сделано разделение для применения преобразования (не одинаковое преобразование для всех сущностей, а разные по значениям этой переменной)
    -   **set_start** - если есть формула, то это флаг на её использование для расчета текущего значения по этой формуле
    -   **formula** - формула, по которой будет вычисляться текущее значение параметра (например, на основе переменных в рабочем датасете ds)

-   ***RHS*** - какие перемнные потенциально будут входить в модель с флагами на включение, с указанием ожидаемых знаков и весов, отражающих "важность" переменных в рамках работы оптимизатора

    -   **token** - имя переменной, которая будет включена в линейную модель
        -   если переменная должна быть преобразована, то к имени переменной через "\_" добавляется имя функции преобразования
        -   переменная в линейной модели модет представлять собой комбинацию, тогда она записывается через формулу как I(), например, I(X1 + X2 + X3), где X1, X2, X3 - переменные в датасете
    -   **fixed** - заготовка на будущее, пока везде ставим 1 (нужна для дальнейшего использования, когда будет добавлен функционал по подбору набора переменных, включаемых в модель, в процессе оптимизации)
    -   **if_active** - флаг на включение переменной в модель (1 - если включаем, 0 - иначе)
    -   **sign** - предпочитаемый знак переменной (1 - если переменная должна быть положительной, -1 - если переменная должна быть отрицательной)
    -   **sign_weight** - вес, отражающий относительную важность знака этой переменной (будет учтено в скорах при оптимизации)
    -   **signif_weight** - вес, отражающий относительную важность значимости этой переменной (будет учтено в скорах при оптимизации)
    -   **display_var** - как переменная должна быть названа в анализе и в репортинге, например, дашборже или при создании презентации (по умолчанию дублируется token; если у нескольких переменных одинаковые имена, они будут агрегированы)

-   ***scoring*** - что входит в оптимизируемую целевую функцию, какие скоры и с какими весами (сами скоры задаются в `fermatrica.evaluation.scoring`)

    При оптимизации максимизируем общий скор, для этого

    -   **rmse** - корень из суммы квадратов остатков - чем меньше, тем лучше - if_invert = 1
    -   **r_squared** - коэффициент детерминации, он же R-квадрат - чем больше тем лучше - if_invert = 0
    -   **r_squared_adjusted** - скорректированный коэффициент детерминации, он же R-квадрат-adjusted - чем больше тем лучше - if_invert = 0
    -   **t_value** - t-статистики тестов на значимость коэффициентов регрессии с учетом весов без учета желаемых знаков, в скор идут все случаи - чем больше, тем лучше - if_invert = 0
    -   **p_value** - p-значения тестов на значимость коэффициентов регрессии с учетом весов без учета желаемых знаков, в скор идут все случаи - чем меньше, тем лучше - if_invert = 1
    -   **sign_simple** - соответствие желаемым знакам с учетом весов, в скор идут случаи несоответствия - чем меньше, тем лучше - if_invert = 1
    -   **sign_t_value** - t-статистики тестов на значимость коэффициентов регрессии с учетом весов и знаков, в скор идут случаи несоответствия знаков желаемым - чем меньше, тем лучше - if_invert = 1
    -   **vif** - среднее значение мер мультиколлинеарности - чем меньше, тем лучше - if_invert = 1
    -   **durbin_watson_score** - статистика теста Дарбина-Уотсона на автокорреляцию в остатках за вычетом 2 (!!! важно, что не durbin_watson) - чем меньше, тем лучше - if_invert = 1

> _Дополнительно:_
> -    <small>Если задан кастомный скоринг в [**custom_scoring**](#ru_cs), его компоненты должны быть заданы на этом листе и активны.</small>
> -    <small>Если хотим перевернуть весь скор (invert), прописываем это при вызове функции оптимизации.</small>

-   ***trans_path_df*** - этот лист будет создан автоматически после первого прогона оптимизатора или может быть исходно создан.

    На этом листе для каждой переменной задается

    -   **variable** - исходная переменная из датасета ds, которая используется в модели,
    -   **variable_fin** - преобразованная переменная, которая используется в модели
    -   **price** - цена за единицу медиа воздействия для каждого медийного канала для корректного отображений кривых (если переменная немедийная, то ячейка с ценой должна быть пустая),
    -   **display_var** - имя переменной, как она должна быть названа в дашборде или при создании презентации.

</details>
<br/>

<details>

Каждое преобразование, заданное на вкладке `params`, создаёт в датасете новую переменную, имя которой составляется из имени исходной переменной и имени функции преобразования, соединённых нижних подчёркиванием. Например, преобразование `softmax` над переменной `tv` создаёт переменную `tv_softmax`.

Преобразования выполняются последовательно.

На той же вкладке задаются коэффициенты для преобразований левой части, они отличаются пустым полем `variable` и значением `coef` в поле `fun`. Коэффициенты применяются в порядке выполнения преобразований левой части.

**<summary>Основные функции, которые используются для преобразований переменных в модели (чтобы узнать подробности, нажмите на треугольник)</summary>**

Функции задаются на листе `params` файла `model_conf.xlsx`.

##### Временные преобразования

***lag()***: лагирование (сдвиг во времени) переменной, запаздывание

-    **n** - число лагов (на сколько периодов сдвиг)
-    **cval** - значение, которым будут заполнены пропуски в начале, которые образуются при сдвиге. По умолчанию 0

***mar()***: правое сглаживание методом скользящих средних

-    **n** - число лагов (на сколько периодов сдвиг)
-    **if_scale** - нужно ли взвешивать сглаженную переменную на среднее значение в трейне + тесте

- ***adstock(), adstockp()***: геометрический адсток (запаздывание, помогает моделировать отложенный эффект). Первый вариант - взвешенный, с суффиксом 'p' - без взвешивания ("pure")

-    **a** - decay rate (какая доля воздействия в период t будет иметь эффект только в t+1)

***adstockpd()***: линейная комбинация двух адстоков с разной скоростью затухания.

-    **a1** - decay rate первого адстока
-    **a2** - decay rate второго адстока
-    **w2** - коэффициент для второго адстока при составлении линейной комбинации (от 0 до 1). Первый адсток взвешивается, соответственно, на `1 - w2`

***dwbl(), dwblp()***: затухание / временное преобразование на базе распределения Вейбулла (второй вариант - "чистый", устойчив к добавлению новых наблюдений в ряду)

-    **dur** - длительность эффекта в периодах (округляется до целого)
-    **shape** - параметр, определяющий форму кривой (shape < 1 — С-образная кривая, shape > 1 — S-образная)
-    **scale** - коэффициент масштаба
-    **norm** - флаг нормализации результата, по умолчанию False

##### Насыщение

***expm1(), expm1scaled()***: экспоненциальное преобразование с вычитанием 1 из результата и взвешиванием с медианой и стандартным отклонением (второй вариант)

-    **product** - множитель для исходной переменной перед взятием экспоненты, делает кривую более гладкой или более пологой

***softmax(), softmaxfull()***: разновидность логистического преобразования (сигмоиды), первый вариант - с вычетом нуля по иксу (чтобы при 0 по X не было значения, отличного от 0, по Y)

-    **lambda** - параметр крутизны преобразования, меньше - круче
-    **avg** - точка перегиба (скорость прироста начинает снижаться)
-    **std** - стандартное отклонение исходной переменной, используется для очищения лямбды от размерности исходной переменной

***logistic()***: разновидность логистического преобразования (сигмоида)

-    **steep** - параметр крутизны преобразования
-    **ec50** - параметр достижения половины уровня насыщения
-    **level** - уровень насыщения (используется для очищения остальных параметров от размерности исходной переменной)

- ***adbudg()***: в зависимости от значений аргументов, приближается к степенной или сигмоидальной форме

-    **steep** - параметр крутизны преобразования
-    **ec50** - параметр достижения половины уровня насыщения
-    **level** - уровень насыщения (используется для очищения остальных параметров от размерности исходной переменной)

...И ещё полдесятка реже используемых функций

##### Комплексные преобразования

***arl(), arlp()***: преобразование, совмещающее насыщение, адсток и запаздывание. В первом случае насыщение задаётся функцией `adbudg()`, во втором - `logistic()`.

-    **a** - decay rate
-    **steep** - параметр крутизны преобразования
-    **ec50** - параметр достижения половины уровня насыщения
-    **level** - уровень насыщения (используется для очищения остальных параметров от размерности исходной переменной)
-    **n_lag** - число лагов (на сколько периодов сдвиг)
-    **cval** - при наличии лага, значение, которым заполняются края смещенного вектора

!   В функциях `arl()` и `arlp()` сначала идёт насыщение, потом адсток, потом сдвиг (лаг)

***price()***: комплексное преобразование с расчётом цен относительно средних по группе в момент времени.

</details>
<br/>

Далее загружаем первичную модель и оцениваем её без использования оптимизатора. Это делается в три шага.

1. **Загрузка модели.** Создаём объект `fermatrica.Model`, считывая `model_conf.xlsx` при инициализации объекта. При более сложной структуре модели на этом же этапе аргументами передаётся adhoc-код и функция построения категориальной модели.
    ```python
    import os
    import fermatrica as fm
   
    pth = os.path.join("code_py", "data", "model_conf.xlsx")
    model = fm.Model(path=pth
                     , custom_predict_fn=None
                     , adhoc_code=[code_py.adhoc.model]
                     , ds=dt_p)
    ```

2. **Выполнение трансформаций внешнего контура.** Построение модели и получение прогноза делается в два шага при помощи одной и той же пары функций. Это необходимо для улучшения производительности у сложных моделей. Сначала выполняются преобразования внешнего контура. Аргумент `set_start` запускает расчёт начальных значений параметров (где таковой задан), относитесь к нему аккуратно.
    ```python
    model = fm.transform(ds=dt_p
                         , model=model
                         , set_start=True
                         , if_by_ref=True)
    ```

3. **Фит внутреннего контура и получение прогноза.** Когда преобразования выполнены, запускаем внутренний контур. На этом шаге строится внутренняя линейная модель, а также туда и обратно совершаются преобразования левой части. Можно запустит `fermatrica.fit_predict()` два раза с разными аргументами - сначала будет построена модель и получен прогноз на трейне, потом получим прогноз уже на всём датасете. Функция `fermatrica.predict_ext()` возвращает прогноз в виде не серии, а фрейма с дополнительными столбцами, удобными для анализа
    ```python
    pred, pred_raw, model = fm.fit_predict(dt_p, model, if_fit=True, return_full=True)

    dt_pred = fm.predict_ext(model, dt_p)
    ```

> _Дополнительно:_  <small>Пример можно посмотреть здесь: [примеры скоро будут готовы]</small>

[Назад к теории - Первичная сборка модели и просмотр результатов](#ru_model_start)

### <a name="ru_step_4"></a> ШАГ 4

Просматриваем результаты, сохраняем, если они нас удовлетворяют

Если хотим сохранить модель, используем метод `fermatrica.model.save` (для неё нужно указать путь к папке, в которую будут сохраняться результаты).

Просмотр результатов возможен

-   в Jupyter Notebook (Блоки **Get results**, **Statistic and metrics**, **Observed and predicted**, **Decomposition and Waterfall**), если модель строится в нём
-   в том же скрипте model.py с использованием написанных в FERMATRICA функций расчета нужных статистик и графиков (тесты при необходимости, график наблюдаемых и прогнозных значений зависимой переменной, график остатков, качество пронноза, декомпозиция). Тогда модель можно не сохранять и продолжать работу над ней, меняя настройки в конфиге и запуская скрипт с оценкой модели
-   в дашборде FERMATRICA_DASH, если модель сохранена

Каждый способ имеет свои плюсы и минусы, исследователь сам может выбрать предпочтительный для него путь.

[Назад к теории - Первичная сборка модели и просмотр результатов](#ru_model_start)

### <a name="ru_step_5"></a> ШАГ 5

Работаем над моделью

#### Меняем параметры и состав переменных вручную

-   Вносим правки в список переменных на листе ***RHS*** в [конфиге](#ru_config) (добавляем новые или "включаем"/"выключаем" уже приписанные)
-   Добавляем/убираем/меняем преобразования на листе ***params***  и в названии переменных, включенных модель на листе ***RHS*** в [конфиге](#ru_config)
-   Меняем значения параметров преобразований в столбце **value** на листе ***params*** в [конфиге](#ru_config)

Оцениваем модель заново, анализируем результаты.

Может проводиться в отдельном скрипте .py или в рамках доработки модели в Jupyter Notebook или в скрипте model.py.

> _Дополнительно:_  <small>Преобразований к одной переменной может быть применено несколько последовательно. Для этого нужно на листе params указать порядок: преобразование *a* накладывается на преобразованную уже переменную X с помощью *b*, тогда в params нужно прописать сначала *b* для X, а потом *a* для X_b, а в модель будет входить X_b_a, что и нужно прописать в **token**</small>

[Назад к теории - Доработка модели](#ru_tuning)

#### Используем оптимизатор для подбора значений параметров

Параметры преобразований могут быть подобраны с использованием численной оптимизации. 

По умолчанию целевая функция (скор) формируется как линейная комбирация метрик, заданных в [конфиге](#ru_config) на листе ***scoring*** в столбце **metrics** и столбце **if_active** (нужные должны быть "включены"), с весами **weight**.
Также целевая функция может быть задана иначе вручную в том жа файле в [**custom_scoring**](#ru_cs).

Поддерживаются локальные и глобальные алгоритмы, на данный момент это локальный алгоритм COBYLA, вызываемый функцией `fermatrica.optimize_local_cobyla()`, и генетический алгоритм PyGAD, вызываемый функцией `fermatrica.optimize_global_ga()`.

<details>

**<summary>Настройки оптимизатора (чтобы узнать подробности, нашмите на треугольник)</summary>**

Для локального алгоритма COBYLA (***local***):

-   *revert_score_sign* - изменять ли знак итогового скора? Меняем, если наш скор максимизируется, потому что COBYLA минимизирует
-   *verbose* - выводить ли диагностическую информацию
-   *epochs* - число эпох - количество раз, которое будет запускаться оптимизатор. В локальном оптимизаторе следующая эпоха наследует предыдущей, поэтому количество оптимальных моделей не зависит от количества эпох и всегда равно 1. Смысл нескольких эпох - "встряхивание" алгоритма (shuffle) для лучшего избегания локальных минимумов
-   *iters_epoch* - максимальное число итераций работы оптимизатора в рамках одной эпохи
-   *error_score* - скор в случае ошибки. Ставим какое-нибудь огромное число, в случае минимизирующего алгоритма - со знаком плюс
-   *ftol_abs* - порог прироста оптимизируемой функции, после которого алгоритм останавливается. COBYLA обращается с ним довольно своенравно, если видит потенциал дальнейшей оптимизации (склон), но не вовсе игнорирует.

Для генетического алгоритма PyGAD (***global***):

-   *revert_score_sign* - изменять ли знак итогового скора? Меняем, если наш скор минимизируется, потому что PyGAD максимизирует
-   *verbose* - выводить ли диагностическую информацию
-   *epochs* - число эпох - количество раз, которое будет запускаться оптимизатор, оно же - количество моделей, которое будет оценено на выходе. В глобальном оптимизаторе эпохи независимы, и в конце каждой сохраняется найденный оптимальный вариант
-   *iters_epoch* - максимальное число итераций работы оптимизатора в рамках одного запуска
-   *pop_size* - размер популяции для генетического алгоритма (исходное число особей-моделей на старте работы алгоритма)
-   *pmutation* - вероятность мутации (случайных изменений в особях-моделях) при генетического алгоритма
-   *max_no_improvement* - максимальное число итераций, при котором нет регистрируемого повышения скора
-   *error_score* - скор в случае ошибки. Ставим какое-нибудь огромное число, в случае максимизирующего алгоритма - со знаком минус
-   *cores* - число логических ядер компьютера, которые могут быть задействованы для расчетов
-   *save_epoch* - путь к папке, в которую сохранять результаты после каждой эпохи

</details>
<br/>

> _Дополнительно:_  <small>Пример можно посмотреть здесь: [coming soon]</small>

Повторяем ШАГ 4 и ШАГ 5, пока не получим результат, который полностью нас удовлетворит.


### <a name="ru_step_6"></a> ШАГ 6

Выгрузка результатов для клиента возможна: 

-   в презентацию PowerPoint
-   в дашборд

> _Дополнительно:_  
> - <small>Дашборд существует в виде отдельного репозитория FERMATRICA_DASH</small>
> - <small>Генератор слайдов входит в состав репозитория FERMATRICA_REP</small>

[Назад к теории - Выгрузка и представление результатов в удобном клиенту формате](#ru_results)

В папке output готовим отчет по модели.

------------------------------------------------------------------------
------------------------------------------------------------------------
## <a name="ru_part_3"></a>Полезные ссылки

- [Документация](https://fermatrica.github.io/docs)
- [Исходный код](https://github.com/FERMATRICA)
- [Примеры](https://github.com/FERMATRICA/fermatrica/tree/master/samples)



