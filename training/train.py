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
from utils import check_file_dir, check_dir, get_augmentation_name_from_config
from attacks.utils import conduct_pgd
from utils.eval_utils import convert_strdiv2num_lst
from augmenter.augmenter import Augmenter
from augment_transforms.trans_utils import get_test_transforms

from inductive_bias_generalisation.utils import measure_inductive_bias_generalisation
from pcstress.basic import default_PCStress


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


def train(n_workers, arch, ds_name, augmentation_config_file, batch_size, num_epochs,
          lr, momentum, weight_decay, milestones, lr_gamma,
          no_wandb, run, model_path, log_grad, log_gradients_freq, eval_epoch_freq,
          watch_mode, pgd_iters, pgd_eps, pgd_alphas, pgd_norms, pgd_targeted, pgd_nontargeted, pgd_random_restarts,
          measure_stress, stress_eps, stress_nsamp,
          inductive_bias_generalisation, inductive_bias_generalisation_n_aug,
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

    test_transforms = get_test_transforms(train_set.augmentation_dict)

    # in test, we don't need config file; hence, set to None.
    test_set = getattr(globals()[DS_MODULE_DICT[ds_name]], DS_CLASS_DICT[ds_name])(DS_DATAROOT,
                                                                                   SPLIT_Train_DICT[
                                                                                       ds_name + '_val'],
                                                                                   None,
                                                                                   device,
                                                                                   transform=test_transforms,
                                                                                   download=True
                                                                                   )

    # noaug_test_set = getattr(globals()['torchvision_datasets'], TORCHDS_CLASS_DICT[ds_name])(DS_DATAROOT,
    #                                                                                    SPLIT_Train_DICT[
    #                                                                                        ds_name + '_val'],
    #                                                                                    download=True,
    #                                                                                    transform=test_transforms)
    #

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=n_workers, worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=n_workers,
                                              worker_init_fn=worker_init_fn)

    augmenter = Augmenter(train_set, device, arch, lr, milestones, lr_gamma, weight_decay, momentum)

    print('start logging with W&B...')
    if not no_wandb:
        # watching the model for logging
        if watch_mode == 'params':
            print('\n\n***************************')
            print('Watching params only ...')
            print('***************************\n\n')
            wandb.watch(augmenter.model, log='parameters', log_freq=log_gradients_freq)
        elif watch_mode == 'all':
            print('\n\n***************************')
            print('Watching all ...')
            print('***************************\n\n')
            wandb.watch(augmenter.model, log='all', log_freq=log_gradients_freq)
        elif watch_mode == 'none':
            print('\n\n***************************')
            print('Watching None ...')
            print('***************************\n\n')
            wandb.watch(augmenter.model, log=None, log_freq=log_gradients_freq)
        elif watch_mode == 'grads':
            print('\n\n***************************')
            print('Watching the Gradients ...')
            print('***************************\n\n')
            wandb.watch(augmenter.model, log="gradients", log_freq=log_gradients_freq)

    print('start training ...')

    # time it for the first epoch
    t_start = time.time()

    for epoch_counter in range(num_epochs):
        count_total, detached_loss_total = 0., 0.
        for batch in tqdm(train_loader, desc='training loop epoch {}'.format(epoch_counter)):
            X, y = batch
            X, y = X.to(device), y.to(device)

            augmenter.update(X, y)

            # log loss
            detached_loss_total += augmenter.get_loss_value()
            count_total += y.size(0)

        # reduce lr
        augmenter.lr_step()
        # increasing epoch counter
        epoch_counter += 1

        # log gradients
        if log_grad and epoch_counter % log_gradients_freq == 0:
            for name, parameter in augmenter.model.named_parameters():
                if parameter.requires_grad:
                    wandb.run.history.torch.log_tensor_stats(parameter.grad.data, "gradients/" + name)

        if epoch_counter % eval_epoch_freq == 0:
            train_count_total, train_correct_total, train_detached_loss_total = 0., 0., 0.
            for train_i, batch in enumerate(
                    tqdm(train_loader, desc='evaluation loop (train) in epoch {}'.format(epoch_counter))):
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
                    tqdm(test_loader, desc='evaluation loop (val) in epoch {}'.format(epoch_counter))):
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
                'epoch:{}\ttrn acc:{:.2f}\ttrn loss:{:.6f}\ttrn loss eval:{:.6f}\ttst acc:{:.2f}\ttst loss:{:.6f}\telapsed:{:.2f}'.format(
                    epoch_counter, train_correct_total / train_count_total * 100,
                                   detached_loss_total / count_total,
                                   train_detached_loss_total / train_count_total,
                                   test_correct_total / test_count_total * 100,
                                   test_detached_loss_total / test_count_total,
                                   time.time() - t_start))

            # logg measures
            if not no_wandb:
                wandb.log({
                    "epoch": epoch_counter,
                    "Train Accuracy": train_correct_total / train_count_total * 100.,
                    "Train Loss": detached_loss_total / count_total,
                    "Train Loss eval": train_detached_loss_total / train_count_total,
                    "Test Accuracy": test_correct_total / test_count_total * 100.,
                    "Test Loss": test_detached_loss_total / test_count_total,
                    "LR": augmenter.optimizer.param_groups[0]['lr']
                })

        # time it for each epoch
        t_start = time.time()

    print('training finished...')
    print('saving the model...')
    # torch.save(augmenter.model.state_dict(), model_path)
    augmenter.save_state(model_path)
    # augmenter.save_augmenter(model_path)
    print('model saved in {}!'.format(model_path))
    # model.load_state_dict(torch.load(model_path))

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
        for attack_norm in pgd_norms:
            for attack_iter in pgd_iters:
                for attack_eps in pgd_eps:
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

    if inductive_bias_generalisation:
        print('start inductive-bias generalisation evaluations ...')
        ibg_results = measure_inductive_bias_generalisation(augmenter, test_set, n_workers, batch_size, worker_init_fn,
                                                            device, inductive_bias_generalisation_n_aug)
        wandb.log(ibg_results)

    if not no_wandb:
        print('saving run id {} ...'.format(run.id))
        model_artifact = wandb.Artifact('model', type='model')
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

        print('saving run id {} ...'.format(run.id))
        config_artifact = wandb.Artifact('augmentation_config_file', type='config')
        config_artifact.add_file(augmentation_config_file)
        run.log_artifact(config_artifact)

        run.join()

    print('FIN!')


if __name__ == "__main__":

    try:
        set_spawn()
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("-n-workers", default=0, type=int, help="number of workers.")
    parser.add_argument("-ds-name", default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'imagenet', 'mnist', 'fmnist', 'kmnist'],
                        help="name of the dataset.")
    parser.add_argument("-arch", default='resnet18', type=str, help="architecture name.")
    parser.add_argument('-training-seed', type=int, default=1, help='random seed for training.')

    # for naming and categorization
    parser.add_argument("-name", default='default name', type=str, help="experiment name.")

    # W&B
    parser.add_argument("-wandb-proj-name", default='data_augmentation_robustness',
                        choices=['data_augmentation_robustness'],
                        help="name of the wandb project.")
    parser.add_argument("-wandb-entity", default='augmentation',
                        choices=['augmentation'],
                        help="name of the wandb entity.")
    parser.add_argument("-group", default='train', type=str,
                        help="experiment group. It will be appended by the name of ds-name and arch.")
    parser.add_argument("-job-type", default='', type=str,
                        help="Experiment job-type. This is used to group results with std in the plots of the web interface."
                             "If not set, the augmentation name is used.")
    parser.add_argument("-tags", nargs='+', default=[], type=str, help="experiment tags.")
    parser.add_argument("-notes", default='', type=str, help="experiment notes.")
    parser.add_argument('--save-code', default=False, action='store_true')
    parser.add_argument('--no-wandb', default=False, action='store_true')

    # logging
    parser.add_argument('-eval-epoch-freq', type=int, default=10, help='frequency of evaluation.')
    parser.add_argument('--log-grad', default=False, action='store_true')
    parser.add_argument('-watch-mode', choices=['none', 'all', 'params', 'grads'], default='none', type=str,
                        help='set watch mode for W&B logging.')
    parser.add_argument('-log-gradients-freq', type=int, default=100, help='frequency of gradient logging.')

    # augmentations
    parser.add_argument('-augmentation-config-file', required=True, type=str,
                        help='path to the yaml file containing the augmentation dictionary.')

    # optimiser
    parser.add_argument("-milestones", nargs='+', default=[80, 160], type=int, help="milestones for lr schedule.")
    parser.add_argument("-lr-gamma", default=0.1, type=float, help="lr gamma for multi-step lr schedule.")
    parser.add_argument("-lr", default=0.1, type=float, help="learning rate.")
    parser.add_argument("-momentum", default=0.9, type=float, help="momentum for SGD.")  # arida:0.1, devs: 0.9
    parser.add_argument("-weight-decay", default=5e-4, type=float, help="number of workers.")  # arida: 5e-4, devs:1e-4
    parser.add_argument("-batch-size", default=128, type=int, help="batch size.")
    parser.add_argument("-num-epochs", default=200, type=int, help="number of epochs.")

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
    parser.add_argument("-pgd-random-restarts", default=3, type=int, help="random restart in PGD attacks.")

    # stress
    parser.add_argument('--measure-stress', default=False, action='store_true')
    parser.add_argument('-stress-eps', nargs='+', type=float, default=[0.001, 0.01, 0.1, 0.25, .5])
    parser.add_argument('-stress-nsamp', type=int, default=1000)

    # inductive bias generalisation
    parser.add_argument('--inductive-bias-generalisation', default=False, action='store_true',
                        help="calculate inductive-bias-generalisation measures.")
    parser.add_argument('-inductive-bias-generalisation-n-aug', default=3, type=int,
                        help="number of times each batch is augmented to be compared with non-augmented minibatch.")

    args = parser.parse_args()

    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)

    augmentation_name = get_augmentation_name_from_config(args.augmentation_config_file)

    hr = '*********************'
    print('\n\n{}\nstart training with uid :{}\nexp name: {} with aug: {}\n{} on {} with {}'.format(hr, uid, args.name,
                                                                                                    augmentation_name,
                                                                                                    hr,
                                                                                                    args.ds_name,
                                                                                                    args.arch))

    if args.job_type == '':
        args.job_type = augmentation_name
        print('job_type has been automatically set to the augmentation_config: {}'.format(augmentation_name))

    # update the args, as we will update W&B with it.
    args.group = "train_{}_{}_{}".format(args.group, args.ds_name, args.arch)
    args.name = '{}_seed{}'.format(args.name, args.training_seed)
    args.tags = ['train'] + args.tags

    if not args.no_wandb:
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
        model_path = os.path.join(PATH_DICT['models'], run.id, 'model.pth')
        check_dir(os.path.join(PATH_DICT['models'], run.id))

        # updating wandb args
        wandb.config.update(args)
        wandb.config.update({'uid': uid})
        wandb.config.update({'model_path': model_path})
    else:
        run = None
        model_path = os.path.join(PATH_DICT['models'], uid, 'model.pth')
        check_dir(os.path.join(PATH_DICT['models'], uid))

    # setting the random seed
    seed_me(args.training_seed)

    arg_vars = vars(args)
    arg_vars['model_path'] = model_path
    arg_vars['run'] = run
    start_time = time.time()

    print('Printing all args:\n')
    print(arg_vars)
    print('\n')

    train(**arg_vars)

    # finishing logging
    if not args.no_wandb:
        run.finish()

    elapsed_time = time.time() - start_time

    print(''.format())
    print('\n\n{}\ntraining finished!\n{}\ntraining time: {}\n\n{}\n{}'.format(hr, hr,
                                                                               time.strftime("%H:%M:%S",
                                                                                             time.gmtime(elapsed_time)),
                                                                               hr, hr))
