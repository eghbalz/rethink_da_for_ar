
import numpy as np
import os

dirname = os.path.dirname(__file__)
path_info = np.loadtxt(os.path.join(dirname, "db.path"), dtype=str, delimiter="\n")

PATH_DICT = {}
for l in path_info:
    db_name, path = l.split(':')
    PATH_DICT[db_name] = path
