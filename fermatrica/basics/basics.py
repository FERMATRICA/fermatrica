"""
Basic / non-specific objects and operations of the FERMATRICA framework core
"""


import pandas as pd
import sys
from line_profiler_pycharm import profile

from fermatrica_utils import DotDict


class FermatricaError(Exception):
    """
    Exception class for errors specific for Fermatrica
    """
    pass


def fermatrica_error(msg: str):
    """
    Raise FermatrricaError exception

    :param msg: error message as string
    :return: void
    """

    sys.tracebacklimit = 1
    raise FermatricaError(msg)

    pass


@profile
def params_to_dict(params: pd.DataFrame) -> "DotDict":
    """
    Converts DataFrame with transformation params to the dictionary
    with elements accessible with dot notation

    :param params: paandas DataFrame with columns `arg` and `value` (at least)
    :return: dictionary
    """

    params_dict = DotDict(dict(zip(params.arg, params.value)))

    return params_dict

