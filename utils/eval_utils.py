
import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.datasets as torchvision_datasets
import time
# import argparse
# import os.path
# import zarr
from tqdm import tqdm
# from datetime import datetime
# import numpy as np
# import random

# logging
import wandb
import numpy as np


def strdiv2num(str_inp):
    """
    does conversion of string division format like 1/255/5 to a float
    :param str:
    :return:
    """
    if type(str_inp) == type(''):
        if '/' in str_inp:
            res = float(str_inp.split('/')[0])
            for i in range(1, len(str_inp.split('/'))):
                tmp = res / float(str_inp.split('/')[i])
                res = tmp
        else:
            res = float(str_inp)
        return res
    else:
        return str_inp


def convert_strdiv2num_lst(str_lst):
    """
    creates a float list from a string of comma separated stringdiv format:
    example:
    '1/255/5,2/255/5'
    :param str_lst:
    :return:
    """
    if type(str_lst) == type(''):
        return [strdiv2num(s) for s in str_lst.split(',')]
    else:
        return str_lst


def evaluate_set(model, test_loader, criterion, device, use_robustness, no_wandb, set_name):
    print('start evaluating on {}...'.format(set_name))
    t = time.time()
    # evaluate
    model.eval_mode_pred()
    with torch.no_grad():
        test_count_total, test_correct_total, detached_test_loss_total = 0., 0., 0.
        try:
            for test_i, batch in enumerate(tqdm(test_loader, desc='testing loop on {}'.format(set_name))):
                X, y = batch
                X, y = X.to(device), y.long().to(device)
                if use_robustness:
                    outputs, _ = model(X, make_adv=False)
                else:
                    outputs = model(X)
                loss = criterion(outputs, y)
                detached_test_loss_total += loss.detach().cpu().numpy().item()
                _, predicted = torch.max(outputs.data, 1)
                test_count_total += y.size(0)
                test_correct_total += predicted.eq(y.data).sum().float().cpu().numpy().item()
        except Exception as e:
            print('error:', e)

        test_acc = test_correct_total / test_count_total * 100.
        test_loss = detached_test_loss_total / test_count_total
        test_time = time.time() - t
        print(
            'tst acc:{:.8f}\ttst loss:{:.8f}\telapsed:{:.8f}'.format(
                test_acc,
                test_loss,
                test_time))
        if not no_wandb:
            wandb.log({
                "Test Accuracy {}".format(set_name): test_acc,
                "Test Loss {}".format(set_name): test_loss,
            })

    print('testing on {} finished...'.format(set_name))
    return test_acc, test_loss, test_time


def sample_zero_centered_spherical(npoints, ndim=784):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def shift_samples(vec, center):
    shifted_vec = vec + center
    return shifted_vec


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig = plt.figure()

    s = sample_zero_centered_spherical(100, 3)
    xi, yi, zi = s
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)

    shifted_s = shift_samples(s, np.array([2, 2, 2])[:, None])
    xi, yi, zi = shifted_s
    ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)

    shifted_s = shift_samples(s, np.array([2, -2, 2])[:, None])
    xi, yi, zi = shifted_s
    ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)

    print('done')
