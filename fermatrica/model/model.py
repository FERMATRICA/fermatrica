"""
FERMATRICA Model defines model with all its complexity.

Model class contains info about the FERMATRICA model serving more as the structure than regular OOP class.
As for now most model pipeline operations are defined as separate functions in other files, while some service
operations are class methods.

It is intentionally to separate data from Model class to make it more flexible and lightweight (even assuming
many operations could be performed with Model without copying and passing by value).

Ideologically FERMATRICA model consists of two layers:
1. Outer layer
2. Inner layer

Inner layer is fast (main) linear model, OLS for time series or slice data and LME for panel data.
Outer layer contains number of components applying to the data before and / or after inner layer. Most
important and frequent components are:
- RHS (X) variable non-linear transformations
- LHS multiplicative transformations
- additional / helper models (category models, cleanse models)

Check other files of the `fermatrica.model` folder / namespace / package to search model pipeline and components
of the Model class.
"""


import datetime
import inspect
import logging
import lzma
import os
import numpy as np
import pandas as pd
import pickle
import re
import zstandard
from typing import Callable

from fermatrica_utils import StableClass, listdir_abs

from fermatrica.basics.basics import fermatrica_error, FermatricaError
from fermatrica.model.model_conf import ModelConf
from fermatrica.model.model_obj import ModelObj, load as model_obj_load


class Model(StableClass):
    """
    Model class contains all the info about the FERMATRICA model. Class consists of two main components:
    1. conf ModelConf object
    2. obj ModelObj object

    ModelConf contains mostly config / data info: LHS definition, RHS definition, params for non-linear
    transformations, model setup etc.

    ModelObj contains mostly Python objects: submodels of different types, function callable objects,
    function and module source code etc.

    These components could be used separately when performance is at stake or large blocks of code to use
    only one part. Nevertheless, the main approach is to use Model object as a whole passing Model and
    not ModelConf or ModelObj as arguments.
    """

    _path: str
    conf: "ModelConf"
    obj: "ModelObj"

    def __init__(
            self,
            path: str | None,
            custom_predict_fn: "Callable | None" = None,
            adhoc_code: list | None = None,
            ds: pd.DataFrame | None = None,
            if_stable: bool = True
    ):
        """
        Initialise class

        :param path: path to the directory with model config XLSX file or directory with full model
        :param custom_predict_fn: function for custom prediction (mostly category model), pass if model
            is created from scratch
        :param adhoc_code: list of loaded Python modules with adhoc code, pass if model is created from scratch
        :param ds: dataset
        :param if_stable: prevent instance from creating new objects after initialising
        """

        if path is not None and not os.path.exists(path):
            fermatrica_error(path + ': path to model definition file or model folder is not found. ' +
                             'To create model from scratch pass `None` as path argument')

        self._path = path

        # two different ways to create Model: from ModelConf only (i.e. from scratch) or load full Model from
        # model directory
        # maybe later: add possibility to create Model programmatically

        if os.path.isfile(self._path):

            self.conf = ModelConf(path=path
                                  , ds=ds
                                  , if_stable=if_stable)

            self.obj = ModelObj(model_conf=self.conf
                                , custom_predict_fn=custom_predict_fn
                                , adhoc_code=adhoc_code
                                , if_stable=if_stable)

        else:
            self._load(ds=ds, if_stable=if_stable)

    def _load(self
              , ds: pd.DataFrame
              , if_stable: bool):
        """
        Load saved model from disc

        :param ds: dataset used to calculate start values in ModelConf etc.
        :param if_stable:
        :return:
        """

        # get specific model paths

        model_files = listdir_abs(self._path)

        # load

        self._load_model_conf(model_files=model_files
                              , ds=ds
                              , if_stable=if_stable)

        self._load_model_obj(model_files=model_files)

    def _load_model_conf(self
                         , model_files: list
                         , ds: pd.DataFrame
                         , if_stable: bool):
        """
        Load saved model config (ModelConf)

        :param model_files: list of files in the model directory
        :param ds: dataset used to calculate start values in ModelConf etc.
        :param if_stable:
        :return:
        """

        model_conf_fl = [x for x in model_files if os.path.basename(x) in ['model_def.xlsx', 'model_conf.xlsx']]

        if len(model_conf_fl) == 1:
            model_conf_fl = model_conf_fl[0]
            self.conf = ModelConf(path=model_conf_fl
                                  , ds=ds
                                  , if_stable=if_stable)
        else:
            msg = "Selected model doesn't contain model definition file"
            fermatrica_error(msg)

    def _load_model_obj(self
                        , model_files: list):
        """
        Load saved model objects (ModelObj)

        :param model_files: list of files in the model directory
        :return:
        """

        model_obj_fl = [x for x in model_files if os.path.basename(x) in ['model_objects.pkl.lzma', 'model_obj.pkl.lzma']]

        if len(model_obj_fl) == 1:
            model_obj_fl = model_obj_fl[0]

            self.obj = model_obj_load(model_obj_fl)
            self.obj.restore_loaded(self.conf)

        else:
            msg = "Selected model doesn't contain model object"
            fermatrica_error(msg)

    def save(self
             , ds: pd.DataFrame | None = None
             , model_rep=None
             , path: str = ''
             , save_format: str = 'XLSX'
             , save_tag: str | None = None
             ):
        """
        Save Model with extra data

        :param ds: dataset
        :param model_rep: ModelRep object used for reporting (see FERMATRICA_REP) or None
        :param path: path to the up-directory, where model directory to be created
        :param save_format: "XLSX" is implemented as for now only
        :param save_tag: specific tag to name model directory; if None, time stamp to be used
        :return:
        """

        # path

        if path == '':
            path = os.path.join(os.getcwd(), 'code_py', 'model_data')

        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            logging.warning('Directory to save model is not found: ' + path +
                            " - and created from scratch. You might delete it by hand if wasn't your intention")

        # model time tag as unique model identifier

        if save_tag is None:
            save_tag = re.sub(' ', '_', datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
            save_tag = re.sub(r':', '', save_tag)

        path = os.path.join(path, 'model_' + save_tag)
        os.makedirs(path, exist_ok=True)

        # save

        self.conf.save(path=path, save_format=save_format)
        self.obj.save(path=path)

        # additional data : if passed

        if model_rep is not None:
            with lzma.open(os.path.join(path, 'model_rep.pkl.lzma'), 'wb') as handle:
                pickle.dump(model_rep, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if ds is not None:
            ds.to_pickle(os.path.join(path, "dt_p.pkl.zstd"), compression="zstd")

        return path


def model_load_ext(model_path: str
                   , missed_stop: bool = False):
    """
    Load saved model with extra elements

    :param model_path: path to the directory with saved model
    :param missed_stop:
    :return:
    """

    rtrn = ''

    # load model

    try:
        model = Model(model_path)
    except FermatricaError as err:
        if missed_stop:
            raise
        else:
            rtrn += str(err)
            model = None

    # get specific model paths

    model_files = listdir_abs(model_path)

    # dt_p (data)

    dt_p_fl = [x for x in model_files if os.path.basename(x) == 'dt_p.pkl.zstd']

    if len(dt_p_fl) == 1:
        dt_p_fl = dt_p_fl[0]

        with zstandard.open(dt_p_fl, 'rb') as handle:
            dt_p = pickle.load(handle)
    elif model is not None:
        dt_p = model.obj.models['main'].model.data.frame

        msg = "data dt_p extracted from model_objects['main']"
        rtrn += "WARNING: " + msg + "\n"
    else:
        dt_p = None

        msg = "data dt_p nor model_objects is provided"
        rtrn += "ERROR: " + msg + "\n"
        if missed_stop:
            fermatrica_error(msg)

    # return data and error/warning statements

    return model, dt_p, rtrn
