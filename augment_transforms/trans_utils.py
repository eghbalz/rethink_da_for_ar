
# import transforms.rand_transforms as transforms
from torchvision import transforms as torch_transforms
import augment_transforms.rand_transforms as deterministic_transforms
import warnings
import torch
import os
import yaml
import sys


# constructing test transforms
# def get_test_transforms(config_dict):
#     test_tranform_lst = []
#     if 'resize' in config_dict['preprocessing'].keys():
#         test_tranform_lst.append(
#             torch_transforms.Resize(config_dict['preprocessing']['resize']['size']))
#     if 'ccrop' in config_dict['preprocessing'].keys():
#         test_tranform_lst.append(
#             torch_transforms.CenterCrop(config_dict['preprocessing']['ccrop']['size']))
#     test_tranform_lst.append(torch_transforms.ToTensor())
#     # test_transforms_sequential = torch.nn.Sequential(test_tranform_lst)
#     # test_transforms = torch.jit.script(test_transforms_sequential)
#
#     test_transforms = torch_transforms.Compose(test_tranform_lst)
#     return test_transforms

def get_test_transforms(conf_dict):
    tranform_lst = []
    if 'preprocessing' in conf_dict.keys():
        print('adding preprocessing transforms ...')
        preprocessing_dict = conf_dict['preprocessing']
        if 'resize' in preprocessing_dict.keys():
            tranform_lst.append(deterministic_transforms.Resize(**preprocessing_dict['resize']))
        if 'ccrop' in preprocessing_dict.keys():
            tranform_lst.append(deterministic_transforms.CenterCrop(**preprocessing_dict['ccrop']))

    # convert to tensor
    tranform_lst.append(deterministic_transforms.ToTensor())

    test_transforms = torch_transforms.Compose(tranform_lst)
    return test_transforms

# return train_trans_dict
def get_fwdtrans_aug(conf_dict):
    # all are applied with probability=1, because the main probability is controlled outside the transform function, and in the forward pass.
    tranform_lst = []
    # tranform_lst.append(transforms.ToPILImage())

    if 'fwdtrans' in conf_dict.keys():
        print('adding augmentation transforms ...')
        augmentation_dict = conf_dict['fwdtrans']['params']
        if 'rcrop' in augmentation_dict.keys():
            print('RandomCrop  selected for transform.')
            tranform_lst.append(torch_transforms.RandomCrop(**augmentation_dict['rcrop']))
        if 'hflip' in augmentation_dict.keys():
            print('RandomHorizontalFlip for transform.')
            # probability in fwdtrans has been taken into account before this stage.
            tranform_lst.append(torch_transforms.RandomHorizontalFlip(p=1.))
        if 'cjitter' in augmentation_dict.keys():
            print('ColorJitter for transform.')
            tranform_lst.append(torch_transforms.ColorJitter(**augmentation_dict['cjitter']))
        if 'rrot' in augmentation_dict.keys():
            print('RandomRotation selected for transform.')
            tranform_lst.append(torch_transforms.RandomRotation(**augmentation_dict['rrot']))
        if 'cutout' in augmentation_dict.keys():
            print('RandomRotation selected for transform.')
            tranform_lst.append(deterministic_transforms.CutoutFWD(p=1.,
                                                                **augmentation_dict['cutout']))

    else:
        print('No transform was selected for training!')

    # convert to tensor
    # tranform_lst.append(transforms.ToTensor())

    # compile the augmentations
    transforms_func = torch.nn.Sequential(*tranform_lst)
    # transforms_func = torch.jit.script(transforms_sequential)

    # transforms = torch_transforms.Compose(tranform_lst)
    return transforms_func


def get_transforms(conf_dict, ds_name):
    tranform_lst = []

    if 'preprocessing' in conf_dict.keys():
        print('adding preprocessing transforms ...')
        preprocessing_dict = conf_dict['preprocessing']
        if 'resize' in preprocessing_dict.keys():
            tranform_lst.append(deterministic_transforms.Resize(**preprocessing_dict['resize']))
        if 'ccrop' in preprocessing_dict.keys():
            tranform_lst.append(deterministic_transforms.CenterCrop(**preprocessing_dict['ccrop']))
    else:
        print('No preprocessing was selected for training!')

    if 'transform' in conf_dict.keys():
        print('adding augmentation transforms ...')
        augmentation_dict = conf_dict['transform']

        if 'pickled_trans' in augmentation_dict.keys():
            print('pickled_trans  selected for transform.')
            tranform_lst.append(deterministic_transforms.PickledTrans(p=augmentation_dict['pickled_trans']['p'],
                                                                      ds_name=ds_name))

        if 'rcrop' in augmentation_dict.keys():
            print('RandomCrop  selected for transform.')
            tranform_lst.append(deterministic_transforms.RandomCrop(p=augmentation_dict['rcrop']['p'],
                                                                    **augmentation_dict['rcrop']['params']))
        if 'hflip' in augmentation_dict.keys():
            print('RandomHorizontalFlip for transform.')
            tranform_lst.append(deterministic_transforms.RandomHorizontalFlip(p=augmentation_dict['hflip']['p']))
        if 'cjitter' in augmentation_dict.keys():
            print('ColorJitter for transform.')
            tranform_lst.append(deterministic_transforms.ColorJitter(p=augmentation_dict['cjitter']['p'],
                                                                     **augmentation_dict['cjitter']['params']))
        if 'rrot' in augmentation_dict.keys():
            print('RandomRotation selected for transform.')
            tranform_lst.append(deterministic_transforms.RandomRotation(p=augmentation_dict['rrot']['p'],
                                                                        **augmentation_dict['rrot']['params']))
        if 'cutout' in augmentation_dict.keys():
            print('RandomRotation selected for transform.')
            tranform_lst.append(deterministic_transforms.Cutout(p=augmentation_dict['cutout']['p'],
                                                                **augmentation_dict['cutout']['params']))
    else:
        print('No transform was selected for training!')

    # convert to tensor
    tranform_lst.append(deterministic_transforms.ToTensor())

    # compile the augmentations
    # transforms_sequential = torch.nn.Sequential(tranform_lst)
    # transforms = torch.jit.script(transforms_sequential)

    transforms_func = torch_transforms.Compose(tranform_lst)

    return transforms_func


def load_augmentation_config(config_file: str) -> dict:
    """Load game config from YAML file."""
    with open(config_file, 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    if sys.version_info < (3, 7):
        warnings.warn('We expect python>3.7')
        assert False

    return config
