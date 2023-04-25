

import torch
import torch.nn as nn
import torchvision

import time
import argparse
import os.path
from datetime import datetime
import numpy as np
import random
from tqdm import tqdm

# logging
import wandb

# we need these for the getattr calls
from datasets import *
from torchvision import datasets as torchvision_datasets

from path_info import PATH_DICT
from utils import check_file_dir, check_dir,get_augmentation_name_from_config
from attacks.utils import conduct_pgd

from augmenter.augmenter import Augmenter
from augment_transforms.trans_utils import get_test_transforms


def set_spawn():
    torch.multiprocessing.set_start_method('spawn')


DS_MODULE_DICT = {
    'cifar10': 'cifar10',
    'cifar100': 'cifar100',
    'svhn': 'svhn',
    'imagenet': 'imagenet',
    'mnist': 'mnist',
    'fmnist': 'fmnist',
    'kmnist': 'kmnist',
}

DS_CLASS_DICT = {
    'cifar10': 'CIFAR10DSET',
    'cifar100': 'CIFAR100DSET',
    'svhn': 'SVHNDSET',
    'imagenet': 'ImageNetDSET',
    'mnist': 'MNISTDSET',
    'fmnist': 'FashionMNISTDSET',
    'kmnist': 'KMNISTDSET',
}

TORCHDS_CLASS_DICT = {
    'cifar10': 'CIFAR10',
    'cifar100': 'CIFAR100',
    'svhn': 'SVHND',
    'imagenet': 'ImageNet',
    'mnist': 'MNIST',
    'fmnist': 'FashionMNIST',
    'kmnist': 'KMNIST',
}
MODELS_MODULE_DICT = {
    'cifar10': 'cifar10_models',
    'cifar100': 'cifar100_models',
    'svhn': 'svhn_models',
    'imagenet': 'imagenet_models',
    'mnist': 'mnist_models',
    'fmnist': 'mnist_models',
    'kmnist': 'mnist_models',
}

SPLIT_Train_DICT = {
    'cifar10_trn': True,
    'cifar100_trn': True,
    'svhn_trn': 'train',
    'imagenet_trn': 'train',
    'mnist_trn': True,
    'fmnist_trn': True,
    'kmnist_trn': True,
    'cifar10_val': False,
    'cifar100_val': False,
    'svhn_val': 'test',
    'imagenet_val': 'val',
    'mnist_val': False,
    'fmnist_val': False,
    'kmnist_val': False,
}


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def seed_me(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate(n_workers, arch, ds_name, augmentation_config_file, batch_size, num_epochs,
             lr, momentum, weight_decay, milestones, lr_gamma, save_dir,augment_testset,
            saving_format,normalize_augmented_data,
             **kwargs):
    device = torch.device("cuda:0")
    DS_DATAROOT = PATH_DICT[ds_name]

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    train_set = getattr(globals()[DS_MODULE_DICT[ds_name]], DS_CLASS_DICT[ds_name])(DS_DATAROOT,
                                                                                    SPLIT_Train_DICT[
                                                                                        ds_name + '_trn'],
                                                                                    augmentation_config_file,
                                                                                    device)

    if augment_testset:
        test_set = getattr(globals()[DS_MODULE_DICT[ds_name]], DS_CLASS_DICT[ds_name])(DS_DATAROOT,
                                                                                    SPLIT_Train_DICT[
                                                                                        ds_name + '_val'],
                                                                                    augmentation_config_file,
                                                                                    device)

    else:
        test_transforms = get_test_transforms(train_set.augmentation_dict)
        test_set = getattr(globals()['torchvision_datasets'], TORCHDS_CLASS_DICT[ds_name])(DS_DATAROOT,
                                                                                       SPLIT_Train_DICT[
                                                                                           ds_name + '_val'],
                                                                                       download=True,
                                                                                       transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                               num_workers=n_workers, worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                                              worker_init_fn=worker_init_fn)

    augmenter = Augmenter(train_set, device, arch, lr, milestones, lr_gamma, weight_decay, momentum)

    print('start training ...')

    # time it for the first epoch
    t_start = time.time()
    file_counter = {}
    for epoch_counter in range(num_epochs):
        for batch in tqdm(train_loader, desc='training loop epoch {}'.format(epoch_counter)):
            X, y = batch
            X, y = X.to(device), y.to(device)

            # generate augmented data
            X_augmented, y_augmented = augmenter.generate_augmented_data(X, y)
            # do what is needed e.g, save
            for img, label, cls in zip(X_augmented, y_augmented, y):
                cls = cls.cpu().numpy().__int__()
                if cls not in file_counter.keys():
                    file_counter[cls] = 0
                    check_dir(os.path.join(save_dir, str(cls)))
                dname = os.path.join(save_dir, str(cls), '{}.{}'.format(file_counter[cls], saving_format))
                lname = os.path.join(save_dir, str(cls), '{}.{}'.format(file_counter[cls], 'lpt'))
                # saving data
                if saving_format == 'jpg':
                    torchvision.utils.save_image(img, dname, normalize=normalize_augmented_data, )
                elif saving_format == 'pt':
                    torch.save(img, dname)
                # saving label
                torch.save(label, lname)

                file_counter[cls] += 1


        # augmenting test data
        if augment_testset:
            for batch in tqdm(test_loader, desc='test loop epoch {}'.format(epoch_counter)):
                X, y = batch
                X, y = X.to(device), y.to(device)

                # generate augmented data
                X_augmented, y_augmented = augmenter.generate_augmented_data(X, y)
                # do what is needed e.g, save
                for img, label, cls in zip(X_augmented, y_augmented, y):
                    cls = cls.cpu().numpy().__int__()
                    if cls not in file_counter.keys():
                        file_counter[cls] = 0
                        check_dir(os.path.join(save_dir, str(cls)))
                    dname = os.path.join(save_dir, str(cls), '{}.{}'.format(file_counter[cls], saving_format))
                    lname = os.path.join(save_dir, str(cls), '{}.{}'.format(file_counter[cls], 'lpt'))
                    # saving data
                    if saving_format == 'jpg':
                        torchvision.utils.save_image(img, dname, normalize=normalize_augmented_data, )
                    elif saving_format == 'pt':
                        torch.save(img, dname)
                    # saving label
                    torch.save(label, lname)

                    file_counter[cls] += 1


        # time it for each epoch
        t_start = time.time()

    print('generation finished...')

    print('FIN!')


if __name__ == "__main__":

    try:
        set_spawn()
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("-n-workers", default=1, type=int, help="number of workers.")
    parser.add_argument("-ds-name", default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'imagenet', 'mnist', 'fmnist', 'kmnist'],
                        help="name of the dataset.")
    parser.add_argument("-arch", default='resnet18', type=str, help="architecture name.")
    parser.add_argument('-training-seed', type=int, default=1, help='random seed for training.')

    # for naming and categorization
    parser.add_argument("-name", default='default name', type=str, help="experiment name.")

    # augmentations
    parser.add_argument('-augmentation-config-file', required=True, type=str,
                        help='path to the yaml file containing the augmentation dictionary.')
    parser.add_argument("-save-dir", default='generated', type=str, help="directory to save augmentations if needed.")
    parser.add_argument('--augment-testset', default=False, action='store_true')
    parser.add_argument("-saving-format", default='jpg',
                        choices=['jpg', 'pt'],
                        help="format for saving the augmented data.")
    parser.add_argument('--normalize-augmented-data', default=False, action='store_true')

    # optimiser
    parser.add_argument("-milestones", nargs='+', default=[80, 160], type=int, help="milestones for lr schedule.")
    parser.add_argument("-lr-gamma", default=0.1, type=float, help="lr gamma for multi-step lr schedule.")
    parser.add_argument("-lr", default=0.1, type=float, help="learning rate.")
    parser.add_argument("-momentum", default=0.9, type=float, help="momentum for SGD.")  # arida:0.1, devs: 0.9
    parser.add_argument("-weight-decay", default=5e-4, type=float, help="number of workers.")  # arida: 5e-4, devs:1e-4

    parser.add_argument("-batch-size", default=128, type=int, help="batch size.")
    parser.add_argument("-num-epochs", default=200, type=int, help="number of epochs.")

    args = parser.parse_args()

    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)

    hr = '*********************'
    print('\n\n{}\nstart training with uid :{}\nexp name: {}\n{} on {} with {}'.format(hr, uid, args.name, hr,
                                                                                       args.ds_name, args.arch))

    # augmentation = os.path.splitext(os.path.basename(args.augmentation_config_file))[0]
    augmentation = get_augmentation_name_from_config(args.augmentation_config_file)
    saving_path = os.path.join(args.save_dir, args.ds_name, augmentation,args.name, uid)
    check_dir(saving_path)

    # setting the random seed
    seed_me(args.training_seed)

    arg_vars = vars(args)
    arg_vars['save_dir'] = saving_path
    start_time = time.time()

    generate(**arg_vars)

    elapsed_time = time.time() - start_time

    print(''.format())
    print('\n\n{}\ngeneration finished!\n{}\ngeneration time: {}\n\n{}\n{}'.format(hr, hr,
                                                                                   time.strftime("%H:%M:%S",
                                                                                                 time.gmtime(
                                                                                                     elapsed_time)),
                                                                                   hr, hr))
