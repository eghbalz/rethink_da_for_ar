
import torch
import torch.nn as nn

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
from pcstress.basic import default_PCStress
from utils import check_file_dir, check_dir, get_augmentation_name_from_config
from utils.eval_utils import convert_strdiv2num_lst
from attacks.utils import conduct_pgd

from augmenter.augmenter import Augmenter
from augment_transforms.trans_utils import get_test_transforms

from inductive_bias_generalisation.utils import measure_inductive_bias_generalisation
from lpips.utils import calc_lpips
from pathlib import Path


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


def evaluate(n_workers, arch, ds_name, augmentation_config_file, batch_size,
             lr, momentum, weight_decay, milestones, lr_gamma,
             no_wandb, run, model_path,
             pgd_iters, pgd_eps, pgd_alphas, pgd_norms, pgd_targeted, pgd_nontargeted, pgd_random_restarts,
             measure_stress, stress_eps, stress_nsamp,
             trained_model_uid, augment_testset,
             inductive_bias_generalisation, inductive_bias_generalisation_n_aug, measure_lpips, lpips_n_aug,
             no_cls_eval, lpips_batch_size, lpips_compare_robust,
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
        # test_set = getattr(globals()['torchvision_datasets'], TORCHDS_CLASS_DICT[ds_name])(DS_DATAROOT,
        #                                                                                    SPLIT_Train_DICT[
        #                                                                                        ds_name + '_val'],
        #                                                                                    download=True,
        #                                                                                    transform=test_transforms)
        # in test, we don't need config file; hence, set to None.
        test_set = getattr(globals()[DS_MODULE_DICT[ds_name]], DS_CLASS_DICT[ds_name])(DS_DATAROOT,
                                                                                       SPLIT_Train_DICT[
                                                                                           ds_name + '_val'],
                                                                                       None,
                                                                                       device,
                                                                                       transform=test_transforms,
                                                                                       download=True
                                                                                       )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                               num_workers=n_workers, worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=n_workers,
                                              worker_init_fn=worker_init_fn)

    augmenter = Augmenter(train_set, device, arch, lr, milestones, lr_gamma, weight_decay, momentum)
    # loading model states.
    augmenter.load_and_set_state(model_path)
    augmenter.set_eval()

    print('start logging with W&B...')
    if not no_wandb:
        wandb.watch(augmenter.model, log=None)

    print('start evaluating ...')

    # time it for the first epoch

    if not no_cls_eval:
        t_start = time.time()
        train_count_total, train_correct_total, train_detached_loss_total = 0., 0., 0.
        for train_i, batch in enumerate(
                tqdm(train_loader, desc='evaluation loop (train)')):
            X, y = batch
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                outputs = augmenter.eval_mode_pred(X)
                trn_loss = augmenter.calc_eval_loss_from_output(outputs, y)
            train_detached_loss_total += trn_loss
            _, predictions = torch.max(outputs.data, 1)
            train_count_total += y.size(0)
            train_correct_total += predictions.eq(y.data).sum().float().cpu().numpy().item()

        test_count_total, test_correct_total, test_detached_loss_total = 0., 0., 0.
        for test_i, batch in enumerate(
                tqdm(test_loader, desc='evaluation loop (val)')):
            X, y = batch
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                outputs = augmenter.eval_mode_pred(X)
                val_loss = augmenter.calc_eval_loss_from_output(outputs, y)
            test_detached_loss_total += val_loss
            _, predictions = torch.max(outputs.data, 1)
            test_count_total += y.size(0)
            test_correct_total += predictions.eq(y.data).sum().float().cpu().numpy().item()

        print(
            'trn acc:{:.2f}\ttrn loss eval:{:.6f}\ttst acc:{:.2f}\ttst loss:{:.6f}\telapsed:{:.2f}'.format(
                train_correct_total / train_count_total * 100,
                train_detached_loss_total / train_count_total,
                test_correct_total / test_count_total * 100,
                test_detached_loss_total / test_count_total,
                time.time() - t_start))

        # logg measures
        if not no_wandb:
            wandb.log({
                "Train Accuracy": train_correct_total / train_count_total * 100.,
                "Train Loss eval": train_detached_loss_total / train_count_total,
                "Test Accuracy": test_correct_total / test_count_total * 100.,
                "Test Loss": test_detached_loss_total / test_count_total,
                "LR": augmenter.optimizer.param_groups[0]['lr']
            })

        print('evaluation finished...')

    print('start PGD attacks ...')

    # targeted attacks
    if pgd_targeted:
        print('start PGD targeted attacks ...')
        # converting the stringdiv format to numeric
        pgd_eps = [convert_strdiv2num_lst(ep) for ep in pgd_eps]
        pgd_alphas = [convert_strdiv2num_lst(alph) for alph in pgd_alphas]

        targetted_attack = True
        attack_losses_targeted, attack_accs_targeted = conduct_pgd(pgd_iters, pgd_eps, pgd_norms, pgd_alphas,
                                                                   pgd_random_restarts,
                                                                   test_loader, augmenter, targetted_attack)
        for attack_norm in pgd_norms:
            for attack_iter in pgd_iters:
                for attack_eps in pgd_eps:
                    attack_loss_targeted = attack_losses_targeted[attack_norm][attack_iter][attack_eps]
                    attack_acc_targeted = attack_accs_targeted[attack_norm][attack_iter][attack_eps]
                    wandb.log({'{} pgd targeted {} {} loss'.format(attack_norm, attack_iter,
                                                                   attack_eps): attack_loss_targeted,
                               '{} pgd targeted {} {} acc'.format(attack_norm, attack_iter,
                                                                  attack_eps): attack_acc_targeted})

    # targeted attacks
    if pgd_nontargeted:
        print('start PGD non-targeted attacks ...')
        # converting the stringdiv format to numeric
        pgd_eps = [convert_strdiv2num_lst(ep) for ep in pgd_eps]
        pgd_alphas = [convert_strdiv2num_lst(alph) for alph in pgd_alphas]

        targetted_attack = False
        attack_losses_nontargeted, attack_accs_nontargeted = conduct_pgd(pgd_iters, pgd_eps, pgd_norms, pgd_alphas,
                                                                         pgd_random_restarts,
                                                                         test_loader, augmenter, targetted_attack)
        for attack_norm_i, attack_norm in enumerate(pgd_norms):
            for attack_iter in pgd_iters:
                for attack_eps in pgd_eps[attack_norm_i]:
                    attack_loss_nontargeted = attack_losses_nontargeted[attack_norm][attack_iter][attack_eps]
                    attack_acc_nontargeted = attack_accs_nontargeted[attack_norm][attack_iter][attack_eps]
                    wandb.log({'{} pgd nontargeted {} {} loss'.format(attack_norm, attack_iter,
                                                                      attack_eps): attack_loss_nontargeted,
                               '{} pgd nontargeted {} {} acc'.format(attack_norm, attack_iter,
                                                                     attack_eps): attack_acc_nontargeted})

    if measure_stress:
        print('start stress calculations ...')
        stress_save_path = model_path.rsplit('/', 1)[0] + f"/b{stress_nsamp}_eps_" + '.'.join(
            [str(ep) for ep in stress_eps]) + "_"
        test_result = default_PCStress(test_loader, augmenter, stress_eps, stress_nsamp)
        torch.save(test_result, stress_save_path + "test_stress.pth")
        # train_result = default_PCStress(train_loader, augmenter, stress_eps, stress_nsamp)
        # torch.save(train_result, stress_save_path + "train_stress.pth")

    if inductive_bias_generalisation:
        print('start inductive-bias generalisation evaluations ...')
        ibg_results = measure_inductive_bias_generalisation(augmenter, test_set, n_workers, batch_size, worker_init_fn,
                                                            device, inductive_bias_generalisation_n_aug)
        wandb.log(ibg_results)

    if measure_lpips:
        print('start lpips evaluations ...')
        lpips_results = calc_lpips(augmenter, test_set, n_workers, batch_size, worker_init_fn,
                                   device, os.path.join(PATH_DICT['models'], trained_model_uid),
                                   lpips_n_aug, lpips_batch_size,
                                   compare_robust=lpips_compare_robust)
        wandb.log(lpips_results)

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
    parser.add_argument('-evaluation-seed', type=int, default=44864, help='random seed for evaluation.')

    # for naming and categorization
    parser.add_argument("-name", default='default name', type=str, help="experiment name.")

    # W&B
    parser.add_argument("-wandb-proj-name", default='data_augmentation_robustness',
                        choices=['data_augmentation_robustness'],
                        help="name of the wandb project.")
    parser.add_argument("-wandb-entity", default='augmentation',
                        choices=['augmentation'],
                        help="name of the wandb entity.")
    parser.add_argument("-group", default='evaluate', type=str,
                        help="experiment group. It will be appended by the name of ds-name and arch.")
    parser.add_argument("-job-type", default='', type=str,
                        help="Experiment job-type. This is used to group results with std in the plots of the web interface."
                             "If not set, the augmentation name is used.")
    parser.add_argument("-tags", nargs='+', default=[], type=str, help="experiment tags.")
    parser.add_argument("-notes", default='', type=str, help="experiment notes.")
    parser.add_argument('--save-code', default=False, action='store_true')
    parser.add_argument('--no-wandb', default=False, action='store_true')

    # augmentations
    parser.add_argument('-augmentation-config-file', default='noaug', type=str,
                        help='path to the yaml file containing the augmentation dictionary.')
    parser.add_argument('--augment-testset', default=False, action='store_true')

    # optimiser
    parser.add_argument("-milestones", nargs='+', default=[80, 160], type=int, help="milestones for lr schedule.")
    parser.add_argument("-lr-gamma", default=0.1, type=float, help="lr gamma for multi-step lr schedule.")
    parser.add_argument("-lr", default=0.1, type=float, help="learning rate.")
    parser.add_argument("-momentum", default=0.9, type=float, help="momentum for SGD.")  # arida:0.1, devs: 0.9
    parser.add_argument("-weight-decay", default=5e-4, type=float, help="number of workers.")  # arida: 5e-4, devs:1e-4
    parser.add_argument("-batch-size", default=100, type=int, help="batch size.")

    # adversarial attacks
    parser.add_argument("--pgd-targeted", default=False, action='store_true')
    parser.add_argument("--pgd-nontargeted", default=False, action='store_true')
    parser.add_argument('-pgd-iters', nargs='+', default=[200], type=int)
    parser.add_argument('-pgd-alphas', nargs='+', default=['0.001/5,0.01/5,0.1/5,0.1/5,0.5/5,1/5',
                                                           '1/255/5,2/255/5,4/255/5,8/255/5,16/255/5,32/255/5'],
                        type=str)
    parser.add_argument('-pgd-eps', nargs='+',
                        default=['0.001,0.01,0.1,0.1,0.5,1', '1/255,2/255,4/255,8/255,16/255,32/255'], type=str)
    parser.add_argument("-pgd-norms", nargs='+', default=['l2', 'linf'], type=str, help="attack norms.")
    parser.add_argument("-pgd-random-restarts", default=0, type=int, help="random restart in PGD attacks.")

    # stress
    parser.add_argument('--measure-stress', default=False, action='store_true')
    parser.add_argument('-stress-eps', nargs='+', type=float, default=[0.001, 0.01, 0.1, 0.1, 0.5, 1, 2])
    parser.add_argument('-stress-nsamp', type=int, default=1000)
    # parser.add_argument("-stress-norms", nargs='+', default=['l2', 'linf'], type=str, help="stress norms.")

    # inductive bias generalisation
    parser.add_argument('--inductive-bias-generalisation', default=False, action='store_true',
                        help="calculate inductive-bias-generalisation measures.")
    parser.add_argument('-inductive-bias-generalisation-n-aug', default=3, type=int,
                        help="number of times each batch is augmented to be compared with non-augmented minibatch.")

    # lpips
    parser.add_argument('--measure-lpips', default=False, action='store_true')
    parser.add_argument('--lpips-compare-robust', default=False, action='store_true')
    parser.add_argument('-lpips-n-aug', default=1, type=int,
                        help="number of times each batch is augmented to be compared with non-augmented minibatch.")
    parser.add_argument('-lpips-batch-size', default=1000, type=int,
                        help="batchsize for lpips.")
    # evaluate
    parser.add_argument('--no-cls-eval', default=False, action='store_true')
    parser.add_argument("-trained-model-uid", default='', type=str, help="uid of the model to be evaluated.")
    parser.add_argument("-trained-model-fullpath", default='', type=str, help="fullpath of the model to be evaluated.")

    args = parser.parse_args()

    if args.trained_model_uid == '' and args.trained_model_fullpath == '':
        raise ValueError('-trained-model-uid or -trained-model-fullpath has to be set in order to load the model.')

    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)

    hr = '*********************'
    print('\n\n{}\nstart training with uid :{}\nexp name: {}\n{} on {} with {}'.format(hr, uid, args.name, hr,
                                                                                       args.ds_name, args.arch))

    augmentation_name = get_augmentation_name_from_config(args.augmentation_config_file)

    if args.job_type == '':
        args.job_type = augmentation_name
        print('job_type has been automatically set to the augmentation_config: {}'.format(augmentation_name))

    if not args.no_wandb:
        # update the args, as we will update W&B with it.
        args.name = '{}_seed{}'.format(args.name, args.evaluation_seed)
        args.group = "eval_{}_{}_{}".format(args.group, args.ds_name, args.arch)
        args.tags = ['eval'] + args.tags

        # initialize wandb for the project
        run = wandb.init(project=args.wandb_proj_name,
                         name=args.name,
                         group=args.group,
                         tags=args.tags,
                         save_code=args.save_code,
                         job_type=args.job_type,
                         notes=args.notes,
                         id=uid,
                         entity=args.wandb_entity,
                         config=vars(args)
                         )
        model_path = os.path.join(PATH_DICT['models'], args.trained_model_uid, 'model.pth')
        # check_dir(os.path.join(PATH_DICT['models'], args.trained_model_uid))

        # updating wandb args
        wandb.config.update(args)
        wandb.config.update({'uid': uid})
        wandb.config.update({'trained_model_uid': args.trained_model_uid})
        wandb.config.update({'trained_model_path': model_path})
    else:
        run = None
        model_path = os.path.join(PATH_DICT['models'], args.trained_model_uid, 'model.pth')
        # check_dir(os.path.join(PATH_DICT['models'], uid))

    if not os.path.exists(model_path):
        raise ValueError('model was not found! :{}'.format(model_path))

    # setting the random seed
    seed_me(args.evaluation_seed)

    arg_vars = vars(args)
    arg_vars['model_path'] = model_path
    arg_vars['run'] = run
    start_time = time.time()

    print('Printing all args:\n')
    print(arg_vars)
    print('\n')

    evaluate(**arg_vars)

    elapsed_time = time.time() - start_time

    print(''.format())
    print('\n\n{}\ntraining finished!\n{}\ntraining time: {}\n\n{}\n{}'.format(hr, hr,
                                                                               time.strftime("%H:%M:%S",
                                                                                             time.gmtime(elapsed_time)),
                                                                               hr, hr))
