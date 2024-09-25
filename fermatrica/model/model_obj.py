"""
FERMATRICA ModelObj is one of two components of Model. ModelObj contains Python objects essential
for the model. As such it differs from ModelConf keeping info related to the model:
scalars, strings, tables.
"""


import copy
import inspect
import logging
import lzma
import os
import pickle
import re
from typing import Callable

from fermatrica_utils import StableClass, import_module_from_string

from fermatrica.basics.basics import fermatrica_error
from fermatrica.model.model_conf import ModelConf
from fermatrica.model.lhs_fun import *


class ModelObj(StableClass):
    """
    ModelObj (model objects) keeps programming / callable objects of the model.
    Use as the component / attribute of `fermatrica.model.model.Model`
    """

    transform_lhs_fn: "Callable | None"
    transform_lhs_src: str | None
    custom_predict_fn: "Callable | None"
    custom_predict_nm: str | None
    models: dict | None
    adhoc_code_src: dict | None

    def __init__(
            self,
            model_conf: "ModelConf",
            custom_predict_fn: "Callable | None" = None,
            adhoc_code: list | None = None,
            if_stable: bool = True
    ):
        """
        Initialise instance

        :param model_conf: ModelConf object (created via Model() before ModelObj)
        :param custom_predict_fn: function for custom prediction (mostly category model), pass if model
            is created from scratch
        :param adhoc_code: list of loaded Python modules with adhoc code, pass if model is created from scratch
        :param if_stable: prevent instance from creating new objects after initialising
        """

        # code objects
        self.adhoc_code_fill(adhoc_code)

        # dictionary for future object models
        self.models = {}

        # LHS transformation function
        self._transform_lhs_generate(model_conf)

        # custom prediction function
        self.custom_predict_fn = custom_predict_fn

        if isinstance(self.custom_predict_fn, Callable):
            self.custom_predict_nm = custom_predict_fn.__module__ + '.' + custom_predict_fn.__name__
        else:
            self.custom_predict_nm = None

        # prevent from creating new attributes
        # use instead of __slots__ to allow dynamic attribute creation during initialisation
        if if_stable:
            self._init_finish()

    def _transform_lhs_generate(self
                                , model_conf: "ModelConf"):
        """
        Generate LHS function from LHS table in ModelConf

        :param model_conf: ModelConf object
        :return:
        """

        if hasattr(model_conf, 'model_lhs') and isinstance(model_conf.model_lhs, pd.DataFrame):
            self.transform_lhs_fn, self.transform_lhs_src = transform_lhs_generate(model_conf.model_lhs)

    def adhoc_code_fill(self
                        , adhoc_code: list):
        """
        Get source code from loaded adhoc Python modules and save them as `adhoc_code_src` attribute

        :param adhoc_code: adhoc Python modules
        :return:
        """

        if adhoc_code is None:
            self.adhoc_code_src = None
        else:
            self.adhoc_code_src = {}
            for obj in adhoc_code:
                self.adhoc_code_src[obj.__name__] = inspect.getsource(obj)

    def restore_loaded(self
                       , model_conf: "ModelConf"):
        """
        When pickling it is important to destroy all callable attributes as non-pickable (and potentially
        non-restorable). When loading callable attributes to be restored from saved source code (by name
        from saved module code or directly from saved function source code)

        :param model_conf: ModelConf object
        :return:
        """

        self._custom_predict_fn_load()
        self._transform_lhs_generate(model_conf)

    def _custom_predict_fn_load(self):
        """
        Restore custom predict function (mostly category model).
        It is important cause function cannot be pickled and restored properly w/o its environment

        """

        if self.custom_predict_nm is not None:
            self.custom_predict_fn = fun_find(fn_name=self.custom_predict_nm, env_code_src=self.adhoc_code_src)

    def save(self
             , path: str = ''
             ):
        """
        Save ModelObj

        :param path: path to the folder
        :return:
        """

        # path

        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            logging.warning('Directory to save model objects is not found: ' + path +
                            " - and created from scratch. You might delete it by hand if wasn't your intention")

        # remove callable objects

        to_save = prepickle(copy.deepcopy(self))

        # save

        with lzma.open(os.path.join(path, 'model_objects.pkl.lzma'), 'wb') as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return path


def transform_lhs_template(ds: pd.DataFrame
                           , y: pd.Series
                           , coef: "DotDict"
                           , to_original):
    """
    Template to dynamically build LTS transform function from LHS settings in ModelConf

    :param ds: dataset
    :param y: target variable
    :param coef: dictionary with coefficients
    :param to_original: direction of the transformation
    :return:
    """

    if to_original:
        tmp_to_original = 1
    else:
        tmp_from_original = 1

    return y


def transform_lhs_empty(ds: pd.DataFrame
                        , y: pd.Series
                        , coef: "DotDict"
                        , to_original: bool):
    """
    Technical function to run program smoothly, if no LHS transformation is required

    :param ds: dataset
    :param y: target variable
    :param coef: dictionary with coefficients
    :param to_original: direction of the transformation
    :return:
    """

    return y


def transform_lhs_generate(fn_src: pd.DataFrame | None):
    """
    Creates LHS transformation function from source

    :param fn_src: pandas DataFrame containing order, type and source code of LHS transformations
    :return:
    """

    if fn_src[fn_src['if_active'] == 1].shape[0] == 0:
        # it is important to run it anyway for global optimizer

        fn_src = inspect.getsourcelines(transform_lhs_empty)[0]
        fn_src[0] = re.sub('transform_lhs_empty', 'transform_lhs_fn', fn_src[0])

        fn_cl = fun_generate(fn_src=fn_src, fn_name='transform_lhs_fn')

        return fn_cl, fn_src

    model_lhs = fn_src[fn_src['if_active'] == 1].copy()

    model_lhs_mult = model_lhs[model_lhs['type'] == "multiplicative"]
    model_lhs_fb = model_lhs[model_lhs['type'] == "free_before"]
    model_lhs_fa = model_lhs[model_lhs['type'] == "free_after"]
    model_lhs_fa1x = model_lhs[model_lhs['type'] == "free_after_1x"]

    fnct_src = inspect.getsourcelines(transform_lhs_template)[0]
    fnct_src[0] = re.sub('transform_lhs_template', 'transform_lhs_fn', fnct_src[0])

    indnt = [x for x in fnct_src if re.search('tmp_to_original = 1', x)][0]
    indnt = re.search(r'^ +', indnt).group(0)

    if model_lhs_fb.shape[0] > 0:

        pos_cur = [i for i in range(len(fnct_src)) if re.search('tmp_to_original = 1', fnct_src[i])][0]
        for ind, row in model_lhs_fb.iterrows():
            fnct_src.insert(pos_cur, indnt + 'y = ' + row['token'] + '\n')

        pos_cur = [i for i in range(len(fnct_src)) if re.search('tmp_from_original = 1', fnct_src[i])][0] + 1
        for ind, row in model_lhs_fb.iterrows():
            fnct_src.insert(pos_cur, indnt + 'y = ' + row['token'] + '\n')

    if model_lhs_mult.shape[0] > 0:

        pos_cur = [i for i in range(len(fnct_src)) if re.search('tmp_to_original = 1', fnct_src[i])][0]
        for ind, row in model_lhs_mult.iterrows():
            fnct_src.insert(pos_cur, indnt + 'y = y / (' + row['token'] + ')\n')

        pos_cur = [i for i in range(len(fnct_src)) if re.search('tmp_from_original = 1', fnct_src[i])][0] + 1
        for ind, row in model_lhs_mult.iterrows():
            fnct_src.insert(pos_cur, indnt + 'y = y * (' + row['token'] + ')\n')

    if model_lhs_fa.shape[0] > 0:

        pos_cur = [i for i in range(len(fnct_src)) if re.search('tmp_to_original = 1', fnct_src[i])][0]
        for ind, row in model_lhs_fa.iterrows():
            fnct_src.insert(pos_cur, indnt + 'y = ' + row['token'] + '\n')

        pos_cur = [i for i in range(len(fnct_src)) if re.search('tmp_from_original = 1', fnct_src[i])][0]
        for ind, row in model_lhs_fa.iterrows():
            fnct_src.insert(pos_cur, indnt + 'y = ' + row['token'] + '\n')

    if model_lhs_fa1x.shape[0] > 0:

        pos_cur = [i for i in range(len(fnct_src)) if re.search('tmp_to_original = 1', fnct_src[i])][0]
        for ind, row in model_lhs_fa1x.iterrows():
            fnct_src.insert(pos_cur, indnt + 'y = ' + row['token'] + '\n')

    fn_src = [x for x in fnct_src if not re.search('tmp_(to|from)_original = 1', x)]
    fn_cl = fun_generate(fn_src=fn_src
                         , fn_name='transform_lhs_fn')

    return fn_cl, fn_src


def fun_find(fn_name: str
             , env_code_src: dict | None):
    """
    Restores custom predict function (mostly category model).

    Function cannot be pickled and restored properly w/o its environment so environment
    to be loaded first.

    :param fn_name: full function name including "path" (module tree with dots)
    :param env_code_src: dictionary of source code of Python modules to look function through
    :return:
    """

    if env_code_src is None or len(env_code_src) == 0:
        fermatrica_error("Cannot restore function " + fn_name + ": adhoc code wasn't saved. " +
                         "Try to re-save model properly")

    for k, v in env_code_src.items():
        fr_cur_name = inspect.currentframe().f_globals['__name__']
        import_module_from_string(k, v, fr_cur_name)

    fn = eval(fn_name)

    return fn


def fun_generate(fn_src: list | str
                 , fn_name: str = 'transform_lhs_fn'):
    """
    Could generate any function from source, but mostly to generate transform_lhs_fn.

    It is important to place it here and not in FERMATRICA_UTILS to add LHS functions
    available in the environment. Another possible approach could be adding import
    inside fun_generate as dynamic code from argument, but could be difficult and complicated
    as well.

    :param fn_src: function source code
    :param fn_name: function name
    :return:
    """

    d = {}

    if not isinstance(fn_src, str):
        fn_src = ''.join(fn_src)

    exec(fn_src, globals(), d)

    return d[fn_name]


def load(path: str) -> ModelObj:
    """
    Load ModelObj from disc

    :param path: path to ModelObj LZMA-archived pickle
    :return:
    """

    with lzma.open(path, 'rb') as handle:
        model_obj = pd.compat.pickle_compat.load(handle)  # use pickle_compat to avoid errors

    return model_obj


def prepickle(model_obj: ModelObj):
    """
    Remove callable objects from ModelObj object before pickling (could be not pickable)

    :param model_obj: ModelObj object
    :return:
    """

    if hasattr(model_obj, 'transform_lhs_fn'):
        model_obj.transform_lhs_fn = None

    if hasattr(model_obj, 'custom_predict_fn'):
        model_obj.custom_predict_fn = None

    return model_obj
