# Marketing Mix Modeling with FERMATRICA

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

