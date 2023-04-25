
import os
from pathlib import Path
import numpy as np
import torch

from PIL import Image
import os
import pickle
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def change_range(tensor, old_min,old_max,new_min,new_max):
    ntensor = (((tensor - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return ntensor


def check_file_dir(directory):
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory)
    if not os.path.exists(directory):
        print('{} not exist. calling mkdir!'.format(directory))
        os.makedirs(directory)

def check_dir(directory):
    if not os.path.exists(directory):
        print('{} not exist. calling mkdir!'.format(directory))
        os.makedirs(directory)

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def write_text(fname, lst):
    with open(fname, 'w') as f:
        for item in lst:
            f.write("%s\n" % item)


def save_numpy(tensor, filename):
    np.save(filename,tensor)


def save_torch(tensor, filename):
    torch.save(tensor, filename)


def get_augmentation_name_from_config(augmentation_config_file):
    augmentation_name = os.path.splitext(os.path.basename(augmentation_config_file))[0]
    return augmentation_name


def pckl_loader(path: str) :#-> Image.Image:
    with open(path, 'rb') as f:
        img = pickle.load(f)
        return img

def pckl_saver(arr,path: str) :
    with open(path, 'wb') as handle:
        pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

def torch_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = torch.load(f)
        return img


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
