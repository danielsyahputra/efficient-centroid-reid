# encoding: utf-8
"""
Partially based on work by:
@author: mikwieczorek

Adapted and extended by:
@author: danielsyahputra
@contact: danielsyahputra
"""

from .df1 import DF1

__factory = {
    "df1": DF1,
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)