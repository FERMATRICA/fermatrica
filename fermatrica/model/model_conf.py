"""
FERMATRICA ModelConf is one of two components of Model. ModelConf keeps info related to the model:
scalars, strings, tables. As such it differs from ModelObj containing Python objects essential
for the model.
"""


import logging
import os
import numpy as np
import pandas as pd
import re
import sys

from line_profiler_pycharm import profile

from fermatrica_utils import pandas_tree_final_child, StableClass, sr_na, fermatrica_warner
from fermatrica.basics.basics import fermatrica_error


class ModelConf(StableClass):
    """
    ModelConf (model config) keeps together setup, LHS, RHS, params transformations etc.
    Use as the component / attribute of `fermatrica.model.model.Model`
    """

    _path: str
    params: pd.DataFrame
    model_rhs: pd.DataFrame
    model_lhs: pd.DataFrame | None
    scoring: pd.DataFrame
    scoring_dict: dict
    trans_path_df: pd.DataFrame | None
    model_type: str | None
    conversion_var: str | None
    summary_type: str
    _model_setup: pd.DataFrame

    def __init__(
            self,
            path: str,
            ds: pd.DataFrame | None = None,
            if_stable: bool = True
    ):
        """
        Initialise instance

        :param path: path to the Excel file containing ModelConf info
        :param ds: dataset to calculate starting values etc.
        :param if_stable: prevent instance from creating new objects after initialising
        """

        if not os.path.isfile(path):
            fermatrica_error(path + ': model definition file not found. Check path and file name')
        self._path = path

        _shts = pd.ExcelFile(self._path)
        shts = _shts.sheet_names
        _shts.close()

        # read LHS
        if 'LHS' in shts:
            self.model_lhs = pd.read_excel(self._path, sheet_name='LHS')
        else:
            self.model_lhs = None

        # read params
        self.params = pd.read_excel(self._path, sheet_name='params')

        if self.params['index_vars'].dtype != "object":
            self.params['index_vars'] = self.params['index_vars'].astype(str)
            self.params.loc[self.params['index_vars'] == 'nan', 'index_vars'] = np.nan

        if self.params['index_aggr'].dtype != "object":
            self.params['index_aggr'] = self.params['index_aggr'].astype(str)
            self.params.loc[self.params['index_aggr'] == 'nan', 'index_aggr'] = np.nan

        if self.params['index_free_var'].dtype != "object":
            self.params['index_free_var'] = self.params['index_free_var'].astype(str)
            self.params.loc[self.params['index_free_var'] == 'nan', 'index_free_var'] = np.nan

        if 'level_0' in self.params.columns:
            del self.params['level_0']

        if ds is not None:
            self.params = params_expand(ds=ds, params=self.params)

        # params checks
        self._check_if_if_active_stable()
        self._check_if_set_start_fixed()
        self._check_if_following_fixed()
        self._check_opt_bounds()

        # read RHS
        self.model_rhs = pd.read_excel(self._path, sheet_name='RHS')

        # find full transformation paths and set marketing prices
        self._trans_path_df_set(shts)

        # read setup
        self._setup(ds)

        # read scoring
        self._scoring_set(shts)

        # model object (empty when created)
        self.model_objects = None

        # finalize LHS, RHS etc.
        self._check_model()

        # prevent from creating new attributes
        # use instead of __slots__ to allow dynamic attribute creation during initialisation
        if if_stable:
            self._init_finish()

    def _setup(self
               , ds: pd.DataFrame | None = None):
        """
        Unfold setup settings to object attributes

        :return:
        """

        model_setup = pd.read_excel(self._path, sheet_name='setup')
        self._model_setup = model_setup.copy()

        if 'target_audience' in model_setup['key'].unique():
            self.target_audience = model_setup[model_setup['key'] == 'target_audience']['value'].astype('str').to_list()

        if 'bs_key' in model_setup['key'].unique():
            self.bs_key = model_setup[model_setup['key'] == 'bs_key']['value'].astype('str').to_list()

        if 'lme_method' in model_setup['key'].unique():
            self.lme_method = model_setup[model_setup['key'] == 'lme_method']['value'].astype('str').to_list()
            
        if 'lme_estimation_method' in model_setup['key'].unique():
            self.lme_estimation_method = model_setup[model_setup['key'] == 'lme_estimation_method']['value'].iloc[0]
            if self.lme_estimation_method not in ['ml', 'reml']:
                fermatrica_error("LME estimation method `" + self.lme_estimation_method + "` not recognized. " +
                                 "Please select one of `ml` or `reml` or remove this setting from the model setup.")
            
        if 'conversion_fun' in model_setup['key'].unique():
            self.conversion_fun = model_setup[model_setup['key'] == 'conversion_fun']['value'].astype('str').to_list()

        if 'exclude_from_curves' in model_setup['key'].unique():
            self.exclude_from_curves = model_setup[model_setup['key'] == 'exclude_from_curves']['value'].astype('str').to_list()

        if 'exclude_curve' in model_setup['key'].unique():
            self.exclude_curve = model_setup[model_setup['key'] == 'exclude_curve']['value'].astype('str').to_list()

        if 'display_var_base' in model_setup['key'].unique():
            self.display_var_base = model_setup[model_setup['key'] == 'display_var_base']['value'].iloc[0]
            self.display_var_base = str(self.display_var_base)

        model_setup = model_setup[~model_setup['key'].isin(['bs_key', 'target_audience', 'lme_method', 'conversion_fun', 'exclude_from_curves', 'exclude_curve', 'display_var_base']) & model_setup['value'].notna()]
        model_setup.apply(lambda x: setattr(self, x['key'], x['value']), axis=1)

        self._check_setup(ds)

        pass

    def _scoring_set(self
                     , shts: list):
        """
        Load standard and custom (if available) scoring

        :param shts:
        :return:
        """

        if 'scoring' in shts:

            self.scoring = pd.read_excel(self._path, sheet_name='scoring')

            tmp = self.scoring[self.scoring['if_active'] == 1]['weight']
            tmp = tmp.div(tmp.sum())
            self.scoring.loc[self.scoring['if_active'] == 1, 'weight'] = tmp

        else:
            self.scoring = pd.DataFrame({'metrics': ['r_squared']
                                         , 'if_invert': [0]
                                         , 'width': [.3]
                                         , 'weight': [1]
                                         , 'if_active': [1]
                                         })

        self.scoring_dict = self.scoring[self.scoring['if_active'] == 1].set_index('metrics').to_dict(orient='index')

        if (hasattr(self, 'custom_scoring')) and (self.custom_scoring != ''):

            score_frm = self.custom_scoring
            for k in self.scoring_dict.keys():
                score_frm = re.sub(r'\b' + k + r'\b', 'scoring_dict["' + k + '"]["value"]', score_frm)

            self.custom_scoring_compiled = compile(score_frm, '<string>', 'eval')

        pass

    def _check_setup(self
                     , ds: pd.DataFrame | None = None):
        """
        Check loaded setup for potential problems

        :return:
        """

        if not hasattr(self, 'Y_var'):
            self.Y_var = 'units'
            fermatrica_warner.warning("Model definition : Setup : `Y_var` value missed, set to default `units`")

        if not hasattr(self, 'price_var'):
            self.price_var = 'price_distr'
            fermatrica_warner.warning("Model definition : Setup : `price_var` value missed, set to default `price_distr`")

        if isinstance(ds, pd.DataFrame) and self.price_var not in ds.columns:
            fermatrica_warner.warning("Model definition : Setup : price variable`" + self.price_var +
                            "` column not found in the dataset, calculations could result in error")

        if not hasattr(self, 'conversion_var') or self.conversion_var == '':
            self.conversion_var = None
            fermatrica_warner.warning("Model definition : Setup : `conversion_var` value missed (it's OK if no conversion is expected)")

        if not hasattr(self, 'model_type') or self.model_type is None or self.model_type == '':
            self.model_type = 'OLS'
            fermatrica_warner.warning("Model definition : Setup : `model_type` value missed, set to default `OLS`")

        if not hasattr(self, 'summary_type') or self.summary_type is None or self.summary_type == '':
            self.summary_type = 'sum'
            fermatrica_warner.warning("Model definition : Setup : `summary_type` value missed, set to default `sum`." +\
                            " Acceptable values: `sum`, `mean`, `fin`, `mean_fin`")
                            
        if not hasattr(self, 'lme_estimation_method') or self.lme_estimation_method is None or self.lme_estimation_method == '':
            self.lme_estimation_method = 'reml'
            fermatrica_warner.warning("Model definition : Setup : `lme_estimation_method` value missed, set to default `reml`." +\
                            " Acceptable values: `ml`, `reml`")

        if not hasattr(self, 'display_var_base') or self.display_var_base is None or self.display_var_base == '':
            self.display_var_base = 'base'
            fermatrica_warner.warning("Model definition : Setup : `display_var_base` value missed, set to default `base`")

        model_type_allowed = ['OLS', 'LME', 'LMEA', 'FE']
        if self.model_type not in model_type_allowed:
            fermatrica_error("Model type `" + self.model_type + "` not recognized. Please select one of " + str(model_type_allowed))

        pass

    def _check_opt_bounds(self):
        """Check the validity of optimization boundaries.

        Parameters
        ----------
        params_subset : pd.DataFrame
            Pandas DataFrame specifying the bounds and starting values for each parameter.

        """

        mask_main = (self.params['if_active'] == 1) & (self.params['fixed'] == 0)
        params_subset = self.params[mask_main].copy()

        # replace empty string in 'lower' and 'upper' with NaN

        params_subset['lower'] = params_subset['lower'].replace('', pd.NA)
        params_subset['upper'] = params_subset['upper'].replace('', pd.NA)

        # set missing bounds to large finite values

        mask = params_subset['lower'].isnull()
        if mask.any():

            # Fill NA in 'variable', 'fun', 'arg' with string 'NA' for display
            missed = (
                params_subset.loc[mask, 'variable'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'fun'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'arg'].fillna('NA').astype(str)
            )
            fermatrica_warner.warning(f"No lower bounds specified for parameters: {missed.tolist()}. Setting to -1e20.")

            params_subset.loc[mask, 'lower'] = -1e20

        mask = params_subset['upper'].isnull()
        if mask.any():

            missed = (
                params_subset.loc[mask, 'variable'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'fun'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'arg'].fillna('NA').astype(str)
            )
            fermatrica_warner.warning(f"No upper bounds specified for parameters: {missed.tolist()}. Setting to +1e20.")

            params_subset.loc[mask, 'upper'] = 1e20

        # check bounds validity

        mask = params_subset['lower'] > params_subset['value']

        if mask.any():
            missed = (
                params_subset.loc[mask, 'variable'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'fun'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'arg'].fillna('NA').astype(str)
            )
            fermatrica_warner.warning(f"Invalid lower bounds for parameters: {missed.tolist()}. Lower bound reset to be less than or equal to starting value.")

            params_subset.loc[mask, 'lower'] = params_subset.loc[mask, 'value']

        mask = params_subset['upper'] < params_subset['value']

        if mask.any():
            missed = (
                params_subset.loc[mask, 'variable'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'fun'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'arg'].fillna('NA').astype(str)
            )
            fermatrica_warner.warning(f"Invalid upper bounds for parameters: {missed.tolist()}. Upper bound reset to be greater than or equal to starting value.")

            params_subset.loc[mask, 'upper'] = params_subset.loc[mask, 'value']

        #

        self.params[mask_main] = params_subset.copy()

    def _check_if_set_start_fixed(self):
        """Check if the parameters with set_start == 1 are fixed."""

        params_subset = self.params.copy()

        mask = (params_subset['set_start'] == 1) & (params_subset['fixed'] == 0)

        if mask.any():

            # Fill NA in 'variable', 'fun', 'arg' with string 'NA' for display
            missed = (
                params_subset.loc[mask, 'variable'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'fun'].fillna('NA').astype(str) + '_' +
                params_subset.loc[mask, 'arg'].fillna('NA').astype(str)
            )
            fermatrica_warner.warning(f"Parameters with set_start == 1 are not fixed: {missed.tolist()}. Setting fixed to 1.")
            params_subset.loc[mask, 'fixed'] = 1

        self.params = params_subset

    def _check_if_following_fixed(self):
        """Check if the parameters, which are supposed to follow other parameters, are fixed."""

        params_subset = self.params.copy()

        mask = (params_subset.if_active == 1) & (params_subset.fun == 'follow_me')

        if mask.any():

            master_params = params_subset[mask][['variable', 'arg', 'value']]

            need_to_be_fixed = []
            for master_feature in master_params.variable.unique():
                master_params_subset = master_params[master_params['variable'] == master_feature]
                who_follows = [x.strip() for x in master_params_subset[master_params_subset.arg == 'who'].value.values[0].split(',')]
                what_follows = master_params_subset[master_params_subset.arg == 'what'].value.values[0].strip()
                which_follows = [x.strip() for x in master_params_subset[master_params_subset.arg == 'which'].value.values[0].split(',')]
                need_to_be_fixed.extend([[x, what_follows, y] for x in who_follows for y in which_follows])

            for [var,fun, arg] in need_to_be_fixed:
                mask = (params_subset['variable'] == var) & (params_subset['fun'] == fun) & (params_subset['arg'] == arg)
                if mask.any():
                    params_subset.loc[mask, 'fixed'] = 1
                    fermatrica_warner.warning(f"Parameter {var}_{fun}_{arg} is set to follow another parameter, but is not fixed. It will be set to fixed automatically.")

        self.params = params_subset

    def _check_if_if_active_stable(self):
        """Check if the transformations have stable if_active, meaning that all the parameters
        of this transformation are either active or inactive."""

        params_subset = self.params[(self.params['variable'].notna()) & (self.params['variable'] != '')].copy()

        unique_transforms = params_subset[['variable','fun']].drop_duplicates()
        unique_transforms['if_active_stable'] = unique_transforms.apply(lambda x: params_subset[(params_subset['variable'] == x['variable']) & (params_subset['fun'] == x['fun'])]['if_active'].nunique() == 1, axis=1)

        mask = (unique_transforms['if_active_stable'] == False)

        if mask.any():

            # Fill NA in 'variable', 'fun', 'arg' with string 'NA' for display
            missed = (
                unique_transforms.loc[mask, 'variable'].fillna('NA').astype(str) + '_' +
                unique_transforms.loc[mask, 'fun'].fillna('NA').astype(str)
            )

            fermatrica_error(f"These transformations have unstable value of 'if_active': {missed.tolist()}. It means that inside one " +
                             f"transformation some parameters are active and some are not, which is not allowed. Please set 'if_active' " +
                             f"for all the parameters the same for each transformation.")


    def _check_model(self):
        """
        Check loaded model for potential problems

        :return:
        """

        # create display_var in RHS if not exists

        if 'display_var' not in self.model_rhs.columns:
            self.model_rhs['display_var'] = self.model_rhs['token']

        # trim tokens (a bit weird, but required so)

        mask = ~sr_na(self.model_rhs['token'])
        self.model_rhs.loc[mask, 'token'] = (self.model_rhs.loc[mask, 'token'].str.
                                             replace(r' *\+ *', ' + ', regex=True))
        self.model_rhs.loc[mask, 'token'] = (self.model_rhs.loc[mask, 'token'].str.
                                             replace(r' *: *', ':', regex=True))

        # trim params

        cols = [x for x in self.params.columns if x in ['variable', 'fun', 'arg']]

        for col in cols:
            mask = ~sr_na(self.params[col])
            self.params.loc[mask, col] = self.params.loc[mask, col].str.strip()

        # set missed display_var in LHS and RHS to token / name

        self.model_rhs.loc[self.model_rhs['display_var'].isna(), 'display_var'] = self.model_rhs.loc[self.model_rhs['display_var'].isna(), 'token']

        if self.model_lhs is not None:
            if 'display_var' not in self.model_lhs.columns:
                self.model_lhs['display_var'] = self.model_lhs['name']
            self.model_lhs.loc[self.model_lhs['display_var'].isna(), 'display_var'] = self.model_lhs.loc[self.model_lhs['display_var'].isna(), 'name']

    def _trans_path_df_set(self
                           , shts: list):
        """
        Check trans_path_df - complex table containing info about transformation chains and media prices

        :param shts:
        :return:
        """

        if 'trans_path_df' in shts:

            self.trans_path_df = pd.read_excel(self._path, sheet_name='trans_path_df')

            if 'display_var' not in self.trans_path_df.columns:
                self.trans_path_df['display_var'] = self.trans_path_df['variable_fin']

        else:

            trans_path_df = self.params[['variable', 'fun']][self.params['variable'].notna() & self.params['if_active'] == 1].drop_duplicates().copy()
            trans_path_df['variable_fin'] = trans_path_df['variable'] + '_' + trans_path_df['fun']

            trans_path_df.drop('fun', axis=1, inplace=True)

            trans_path_df = pandas_tree_final_child(trans_path_df, 'variable', 'variable_fin')
            trans_path_df['price'] = 1

            trans_path_df['display_var'] = trans_path_df['variable_fin']

            self.trans_path_df = trans_path_df

        pass

    def save(self
             , path: str = ''
             , save_format: str = 'XLSX'
             ):
        """
        Save ModelConf

        :param path: path to the directory to save
        :param save_format: only "XLSX" is supported as for now
        :return:
        """

        # path

        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            logging.warning('Directory to save the model configuration is not found: ' + path +
                            " - and created from scratch. You might delete it by hand if it wasn't your intention")

        # list of data frames to save

        params = self.params.copy()

        if 'index' in params.columns.tolist():
            del params['index']
        if 'level_0' in params.columns.tolist():
            del params['level_0']

        model_conf_list = {
            'params': params
            , 'RHS': self.model_rhs
            , 'setup': self._model_setup
            , 'scoring': self.scoring
        }

        if self.model_lhs is not None:
            model_conf_list['LHS'] = self.model_lhs

        if self.trans_path_df is not None:
            model_conf_list['trans_path_df'] = self.trans_path_df

        # format-specific savers

        if save_format in ['XLSX']:

            with pd.ExcelWriter(os.path.join(path, 'model_conf.xlsx'), engine='xlsxwriter') as writer:
                for df_name, df in model_conf_list.items():
                    # Write dataframe to Excel
                    df.to_excel(writer, sheet_name=df_name, index=False, freeze_panes=(1, 0))
                    
                    # Get the xlsxwriter workbook and worksheet objects
                    workbook = writer.book
                    worksheet = writer.sheets[df_name]
                    
                    # Auto-adjust column widths based on content (exclude last column)
                    for idx, col in enumerate(df.columns):
                        # Skip auto-adjustment for the last column
                        if idx == len(df.columns) - 1:
                            continue
                            
                        # Calculate max width needed for the column
                        max_len = max(
                            df[col].astype(str).str.len().max(),  # Max content length
                            len(str(col))  # Header length
                        )
                        # Set column width with some padding (minimum 10, maximum 50)
                        adjusted_width = min(max(max_len + 2, 10), 50)
                        worksheet.set_column(idx, idx, adjusted_width)
                    
                    # Add autofilter to the data range
                    if len(df) > 0:
                        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

        else:
            fermatrica_error('Saving error. ' + save_format + ' format is not yet implemented. ' +
                             ' Please select one of the following formats: XLSX.')

        return path


@profile
def params_expand(ds: pd.DataFrame
                  , params: pd.DataFrame):
    """
    Cleanse transformation params table and expand if different params per group are expected.

    :param ds: dataset
    :param params: params DataFrame
    :return:
    """

    params = params.copy()
    params.reset_index(inplace=True)

    # remove empty lines
    params = params[(params['variable'].notna()) | (params['fun'].notna())]

    # set not to tune non-numeric variables
    params.loc[params['type'].isin(['str', 'bool']), 'fixed'] = 1

    params.loc[params['arg'].isna(), 'arg'] = 'dummy'
    params.loc[params['arg'].isin(['dummy']), 'fixed'] = 1

    # other types

    if 'float64' in params['type'].tolist():
        params.loc[params['type'] == 'float64', 'value'] = np.float64(params.loc[params['type'] == 'float64', 'value'])

    if 'bool' in params['type'].tolist():
        params.loc[params['type'] == 'bool', 'value'] = params.loc[params['type'] == 'bool', 'value'] \
            .replace({'True': True, '1': True, 1: True, 'False': False, '0': False, 0: False})

    # expand

    params_subset = params[(params['if_active'] == 1) & (params['index_free_var'].notna()) &
                           ~(params['index_free_var'].str.contains('___', regex=True, na=False))].copy()

    if params_subset.shape[0] == 0:
        return params

    try:
        params_subset['index_free_var'] = params_subset['index_free_var'].apply(
            lambda x: [x + '___' + y for y in ds[ds['listed'] == 2][x].unique()])
    except TypeError as e:
        fermatrica_error(f"You might have non-string values in index_free_var column. It is common when using 'bs_key' or something. " + 
                         f"Please make 'bs_key' strings, i.e. adding prefix like 'b', or use another variable")
        return params

    params_subset = params_subset.explode('index_free_var').reset_index(drop=True)

    params = pd.concat([params[~(params['index'].isin(params_subset['index']))], params_subset], ignore_index=True).sort_values(by=['index', 'index_free_var'])
    params = params.drop('index', axis=1).reset_index().drop('index', axis=1)

    return params


