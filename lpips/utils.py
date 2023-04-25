
import copy
import torch
from tqdm import tqdm
from augmenter.augmenter import Augmenter
from augment_transforms.trans_utils import load_augmentation_config
import numpy as np
import lpips
from torch.utils.data import TensorDataset, DataLoader

import os
import pickle

MAX_DISTANCE = 999


# loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization


def normalise(img):
    img *= 2.  # in [0, 2]
    img -= 1.  # in [-1, 1]
    return img


def train_PCA(A, k=100, niter=5, center=False):
    """

    :param A: m_files x n_feature
    :param rank:
    :return:
    """
    m = A.mean(0)
    (U, S, V) = torch.pca_lowrank(A - m, q=k, center=center, niter=niter)
    return V, m


def apply_PCA(V, m, A, k=100):
    return torch.matmul(A - m, V[:, :k])


FEAT_DIM = 124928


def encode_vgg(set_target, loss_fn, batch_size=1000):
    target_t = TensorDataset(set_target.cpu())
    target_l = DataLoader(target_t, batch_size=batch_size)

    embeddings = torch.empty((batch_size * target_l.__len__(), FEAT_DIM))
    ix = 0
    for x_target in tqdm(target_l, desc='vgg encoding loop '):
        with torch.no_grad():
            outs = loss_fn.net.forward(2 * torch.cat(x_target).cuda() - 1.)
            emb = []
            for o in outs:
                emb.append(lpips.normalize_tensor(o).flatten(1))
            emb = torch.hstack(emb)
            embeddings[ix:ix + batch_size, :] = emb
            ix += batch_size
    return embeddings


def emb_pdistance(emb1, emb2):
    pdist = torch.nn.PairwiseDistance(p=2)
    distances = []
    for emb in tqdm(emb1, desc='calc pairwise dist'):
        dist = pdist(emb.repeat(emb2.shape[0], 1), emb2)
        distances.append(dist)
    return torch.vstack(distances)


# def calc_pairwise_distance(set_target, set_all, loss_fn, batch_size=1000):
#     target_t = TensorDataset(set_target.cpu())
#     all_t = TensorDataset(set_all.cpu())
#     target_l = DataLoader(target_t, batch_size=batch_size)
#     all_l = DataLoader(all_t, batch_size=batch_size, pin_memory=True, num_workers=4)
#
#     distances = []
#     for x_target in tqdm(target_l, desc='sample loop target'):
#         distances_sample = []
#         for b_all in tqdm(all_l, desc='batch loop all'):
#             with torch.no_grad():
#                 # d = loss_fn.forward(x_target[0].cuda(), torch.cat(b_all).cuda(), normalize=True, )
#                 # lpips.normalize_tensor(outs0[-1])
#                 d = loss_fn.forward(x_target[0].cuda(), torch.cat(b_all).cuda(), normalize=True, )
#                 distances_sample.append(d.flatten())
#         distances.append(torch.cat(distances_sample))
#     return torch.vstack(distances)
#
#
# def knn(dists, y_ix):
#     dists_all_sorted_ix = torch.argsort(dists, dim=-1, descending=False)
#     dists_nearest_neighbor_ix = dists_all_sorted_ix[:, 0]
#
#     counter = {'trn_aug': 0.,
#                'trn_noaug': 0.,
#                'tst_aug': 0.,
#                'tst_noaug': 0.,
#                'other': 0.,
#                'sum': 0.
#                }
#     for ix in dists_nearest_neighbor_ix:
#         counter['sum'] += 1.
#         if ix >= y_ix[0] and ix < y_ix[0]:
#             counter['trn_aug'] += 1.
#         elif ix >= y_ix[1] and ix < y_ix[2]:
#             counter['trn_noaug'] += 1.
#         elif ix >= y_ix[2] and ix < y_ix[3]:
#             counter['tst_aug'] += 1.
#         elif ix >= y_ix[3]:
#             counter['tst_noaug'] += 1.
#         else:
#             counter['other'] += 1.
#     print('KNN results:{}'.format(counter))
#     return counter
from torch.utils.data import Sampler, SubsetRandomSampler
from datasets.advcifar10 import AdvCIFAR10DSET
from path_info import PATH_DICT


def calc_lpips(augmenter, test_set, n_workers, batch_size,
               worker_init_fn, device, save_dir, n_augment=1, lpips_batch_size=1000,
               compare_robust=False):
    """

    :param augmenter:
    :param test_set:
    :param n_workers:
    :param batch_size:
    :param worker_init_fn:
    :param device:
    :param save_dir:
    :param n_augment:
    :param lpips_batch_size:
    :return:
    """

    # loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
    loss_fn_vgg = loss_fn_vgg.to(device)
    # set all aug. probabilities to 1.
    print('using the training augmentation_config_file')
    augmentation_dict_ = copy.deepcopy(augmenter.dset.augmentation_dict)

    # updating config with probability 1. in all augmentations
    for k_aug in augmentation_dict_.keys():
        if k_aug == 'transform':
            for k_trans in augmentation_dict_[k_aug].keys():
                augmentation_dict_[k_aug][k_trans]['p'] = 1.
        elif k_aug == 'fwd-pass':
            for k_fwdpass in augmentation_dict_[k_aug].keys():
                if augmentation_dict_[k_aug][k_fwdpass].get('p'):
                    augmentation_dict_[k_aug][k_fwdpass]['p'] = 1.
    # no aug. transform
    noaug_transform = test_set.transform

    # copy train set
    train_set_aug = copy.deepcopy(augmenter.dset)
    train_idx = np.random.choice(range(train_set_aug.__len__()), 10000, replace=False)
    train_sampler = SubsetRandomSampler(train_idx)
    # train_set_aug = torch.utils.data.Subset(train_set_aug, subsample)

    # reset dset transform with new config
    train_set_aug.reset_augdict_transform(augmentation_dict_)

    # prep. noaug train dset
    train_set_noaug = copy.deepcopy(train_set_aug)
    # train_set_noaug = torch.utils.data.Subset(train_set_noaug, subsample)
    # train_set_noaug.transform = noaug_transform
    train_set_noaug.set_transform(noaug_transform)

    # prep. augmenter object
    augmenter_ = Augmenter(train_set_aug, device, augmenter.arch, augmenter.lr, augmenter.milestones,
                           augmenter.lr_gamma, augmenter.weight_decay, augmenter.momentum)
    augmenter_.set_model_states(augmenter.model.state_dict())

    # re-init dicts
    augmenter_.init_dicts()

    # reset required params
    augmenter_.set_fwdpass_aug_dict()

    # reset fwd transforms
    augmenter_.set_fwdpass_transforms()

    # aug transform
    aug_transform = augmenter_.dset.transform

    # reset loaders
    test_set_aug = copy.deepcopy(test_set)
    # test_set_aug.transform = aug_transform
    test_set_aug.set_transform(aug_transform)

    # prep. noaug test dset
    test_set_noaug = copy.deepcopy(test_set_aug)
    # test_set_noaug.transform = noaug_transform
    test_set_noaug.set_transform(noaug_transform)

    if compare_robust:
        train_adv_set_robust = AdvCIFAR10DSET(PATH_DICT['advcifar10'], augmentation_config_file=None,
                                              transform=noaug_transform, device=device,
                                              db_name='robust')
        # adv_set_robust.reset_augdict_transform(noaug_transform)

        train_adv_set_nonrobust = AdvCIFAR10DSET(PATH_DICT['advcifar10'], augmentation_config_file=None,
                                                 transform=noaug_transform, device=device, db_name='nonrobust')
        # adv_set_nonrobust.reset_augdict_transform(noaug_transform)

        robust_idx = np.random.choice(range(train_adv_set_robust.__len__()), 10000, replace=False)
        robust_sampler = SubsetRandomSampler(robust_idx)
        train_adv_set_robust_loader = torch.utils.data.DataLoader(train_adv_set_robust,
                                                                  batch_size=batch_size, shuffle=False,
                                                                  num_workers=n_workers,
                                                                  sampler=robust_sampler)

        nonrobust_idx = np.random.choice(range(train_adv_set_nonrobust.__len__()), 10000, replace=False)
        nonrobust_sampler = SubsetRandomSampler(nonrobust_idx)
        train_adv_set_nonrobust_loader = torch.utils.data.DataLoader(train_adv_set_nonrobust, batch_size=batch_size,
                                                                     shuffle=False,
                                                                     num_workers=n_workers,
                                                                     sampler=nonrobust_sampler)

    train_loader_aug = torch.utils.data.DataLoader(train_set_aug, batch_size=batch_size, shuffle=False,
                                                   num_workers=n_workers, worker_init_fn=worker_init_fn,
                                                   sampler=train_sampler)
    test_loader_aug = torch.utils.data.DataLoader(test_set_aug, batch_size=batch_size, shuffle=False,
                                                  num_workers=n_workers)

    train_loader_noaug = torch.utils.data.DataLoader(train_set_noaug, batch_size=batch_size, shuffle=False,
                                                     num_workers=n_workers, worker_init_fn=worker_init_fn,
                                                     sampler=train_sampler)
    test_loader_noaug = torch.utils.data.DataLoader(test_set_noaug, batch_size=batch_size, shuffle=False,
                                                    num_workers=n_workers)

    prev_training = bool(augmenter.model.training)
    augmenter.set_eval()

    if compare_robust:
        # get test predictions for noaug
        train_robust_set = []
        for test_i, batch in enumerate(
                tqdm(train_adv_set_robust_loader, desc='LPIPS data prep loop (robust)')):
            X, y = batch
            X, y = X.to(device), y.to(device)
            for augment_i in range(n_augment):
                train_robust_set.append(X)

        train_robust_set = torch.cat(train_robust_set)

        train_nonrobust_set = []
        for test_i, batch in enumerate(
                tqdm(train_adv_set_nonrobust_loader, desc='LPIPS data prep loop (nonrobust)')):
            X, y = batch
            X, y = X.to(device), y.to(device)
            for augment_i in range(n_augment):
                train_nonrobust_set.append(X)

        train_nonrobust_set = torch.cat(train_nonrobust_set)

    # get train predictions for aug
    train_aug = []
    for train_i, batch in enumerate(
            tqdm(train_loader_aug, desc='LPIPS data prep loop (train noaug) ')):
        X, y = batch
        X, y = X.to(device), y.to(device)
        for augment_i in range(n_augment):
            with torch.no_grad():
                x_aug, _ = augmenter.generate_augmented_data(X, y)
                train_aug.append(x_aug)
    train_aug = torch.cat(train_aug)
    # get train predictions for noaug
    train_noaug = []
    for train_i, batch in enumerate(
            tqdm(train_loader_noaug, desc='LPIPS data prep loop (train aug) ')):
        X, y = batch
        X, y = X.to(device), y.to(device)
        for augment_i in range(n_augment):
            # with torch.no_grad():
            #     x_aug, _ = augmenter.generate_augmented_data(X, y)
            train_noaug.append(X)

    train_noaug = torch.cat(train_noaug)
    # # get test predictions for aug
    # test_aug = []
    # for test_i, batch in enumerate(
    #         tqdm(test_loader_aug, desc='LPIPS data prep loop (val aug)')):
    #     X, y = batch
    #     X, y = X.to(device), y.to(device)
    #     for augment_i in range(n_augment):
    #         with torch.no_grad():
    #             # get augmented output
    #             x_aug, _ = augmenter.generate_augmented_data(X, y)
    #             test_aug.append(x_aug)
    # test_aug = torch.cat(test_aug)

    # get test predictions for noaug
    test_noaug = []
    for test_i, batch in enumerate(
            tqdm(test_loader_noaug, desc='LPIPS data prep loop (val noaug)')):
        X, y = batch
        X, y = X.to(device), y.to(device)
        for augment_i in range(n_augment):
            # with torch.no_grad():
            # x_aug, _ = augmenter.generate_augmented_data(X, y)
            test_noaug.append(X)

    test_noaug = torch.cat(test_noaug)

    print('train set encoding...')
    # subsample = np.random.choice(range(train_noaug.__len__()), 10000, replace=False)
    # train_noaug_10k = train_noaug[subsample]
    train_noaug_emb = encode_vgg(train_noaug, loss_fn_vgg, batch_size=lpips_batch_size)
    test_noaug_emb = encode_vgg(test_noaug, loss_fn_vgg, batch_size=lpips_batch_size)

    print('PCA training...')
    pca_train_data = torch.cat((train_noaug_emb, test_noaug_emb))
    V, m = train_PCA(pca_train_data, 100)

    train_noaug_emb_pca = apply_PCA(V, m, train_noaug_emb, 100)
    test_noaug_emb_pca = apply_PCA(V, m, test_noaug_emb, 100)

    print('test set encoding...')
    # subsample = np.random.choice(range(train_aug.__len__()), 10000, replace=False)
    # train_aug_10k = train_aug[subsample]
    train_aug_emb = encode_vgg(train_aug, loss_fn_vgg, batch_size=lpips_batch_size)
    # test_aug_emb = encode_vgg(test_aug, loss_fn_vgg, batch_size=lpips_batch_size)
    train_aug_emb_pca = apply_PCA(V, m, train_aug_emb, 100)
    # test_aug_emb_pca = apply_PCA(V, m, test_aug_emb, 100)

    if compare_robust:
        print('robust set encoding...')
        # train_robust_emb = encode_vgg(train_robust_set, loss_fn_vgg, batch_size=lpips_batch_size)
        print('nonrobust set encoding...')
        train_nonrobust_emb = encode_vgg(train_nonrobust_set, loss_fn_vgg, batch_size=lpips_batch_size)

        # train_robust_emb_pca = apply_PCA(V, m, train_robust_emb, 100)
        train_nonrobust_emb_pca = apply_PCA(V, m, train_nonrobust_emb, 100)

        # orig_data_pca = torch.cat((train_noaug_emb_pca, test_noaug_emb_pca))
        # dist_train = emb_pdistance(train_aug_emb_pca, train_noaug_emb_pca)
        # dist_test = emb_pdistance(train_aug_emb_pca, test_noaug_emb_pca)
        # dist_self = emb_pdistance(train_aug_emb_pca, train_aug_emb_pca)
        # dist_self = torch.Tensor.fill_diagonal_(dist_self, MAX_DISTANCE, wrap=False)
        ### dist_augtrn = torch.cdist(test_aug_emb_pca, train_aug_emb_pca, compute_mode='use_mm_for_euclid_dist_if_necessary')
        dist_train_robexp = torch.cdist(train_aug_emb_pca, train_noaug_emb_pca,
                                 compute_mode='use_mm_for_euclid_dist_if_necessary')
        # dist_robust = torch.cdist(train_aug_emb_pca, train_robust_emb_pca,
        #                           compute_mode='use_mm_for_euclid_dist_if_necessary')
        dist_nonrobust_robexp = torch.cdist(train_aug_emb_pca, train_nonrobust_emb_pca,
                                     compute_mode='use_mm_for_euclid_dist_if_necessary')
        dist_self_robexp = torch.cdist(train_aug_emb_pca, train_aug_emb_pca,
                                    compute_mode='use_mm_for_euclid_dist_if_necessary')
        dist_self_robexp = torch.Tensor.fill_diagonal_(dist_self_robexp, MAX_DISTANCE, wrap=False)

        # min_dist_robust, min_dist_robust_ix = dist_robust.min(1)
        min_dist_nonrobust_robexp, min_dist_nonrobust_ix_robexp = dist_nonrobust_robexp.min(1)
        min_dist_train_robexp, min_dist_train_ix_robexp = dist_train_robexp.min(1)
        min_dist_self_robexp, min_dist_self_rob_ix_robexp = dist_self_robexp.min(1)

        min_all_robexp = torch.vstack([min_dist_nonrobust_robexp,
                                       min_dist_train_robexp,
                                       min_dist_self_robexp])

        target_ix_robexp = torch.argmin(min_all_robexp, 0)

        # robust_perc_robexp = torch.where(target_rob_ix == 0)[0].shape[0] / target_rob_ix.shape[0]
        nonrobust_perc_robexp = torch.where(target_ix_robexp == 0)[0].shape[0] / target_ix_robexp.shape[0]
        train_perc_robexp = torch.where(target_ix_robexp == 1)[0].shape[0] / target_ix_robexp.shape[0]
        self_perc_robexp = torch.where(target_ix_robexp == 2)[0].shape[0] / target_ix_robexp.shape[0]

    # orig_data_pca = torch.cat((train_noaug_emb_pca, test_noaug_emb_pca))
    # dist_train = emb_pdistance(train_aug_emb_pca, train_noaug_emb_pca)
    # dist_test = emb_pdistance(train_aug_emb_pca, test_noaug_emb_pca)
    # dist_self = emb_pdistance(train_aug_emb_pca, train_aug_emb_pca)
    # dist_self = torch.Tensor.fill_diagonal_(dist_self, MAX_DISTANCE, wrap=False)
    # dist_augtrn = torch.cdist(test_aug_emb_pca, train_aug_emb_pca, compute_mode='use_mm_for_euclid_dist_if_necessary')

    dist_train = torch.cdist(train_aug_emb_pca, train_noaug_emb_pca,
                             compute_mode='use_mm_for_euclid_dist_if_necessary')
    dist_test = torch.cdist(train_aug_emb_pca, test_noaug_emb_pca,
                            compute_mode='use_mm_for_euclid_dist_if_necessary')
    dist_self = torch.cdist(train_aug_emb_pca, train_aug_emb_pca,
                            compute_mode='use_mm_for_euclid_dist_if_necessary')
    dist_self = torch.Tensor.fill_diagonal_(dist_self, MAX_DISTANCE, wrap=False)
    # dist_augtrn = torch.cdist(test_aug_emb_pca, train_aug_emb_pca, compute_mode='use_mm_for_euclid_dist_if_necessary')

    min_dist_train, min_dist_train_ix = dist_train.min(1)
    min_dist_test, min_dist_test_ix = dist_test.min(1)
    min_dist_self, min_dist_self_ix = dist_self.min(1)

    # min_all = torch.cat([min_dist_train, min_dist_test, min_dist_self])
    min_all = torch.vstack([min_dist_train, min_dist_test, min_dist_self])

    target_ix = torch.argmin(min_all, 0)

    train_perc = torch.where(target_ix == 0)[0].shape[0] / target_ix.shape[0]
    test_perc = torch.where(target_ix == 1)[0].shape[0] / target_ix.shape[0]
    self_perc = torch.where(target_ix == 2)[0].shape[0] / target_ix.shape[0]

    # if compare_robust:
    #     min_all_combined = torch.vstack([min_dist_train, min_dist_test, min_dist_self, min_dist_robust, min_dist_nonrobust])
    #
    #     target_ix_combined = torch.argmin(min_all_combined, 0)
    #
    #     train_perc_combined = torch.where(target_ix_combined == 0)[0].shape[0] / target_ix_combined.shape[0]
    #     test_perc_combined = torch.where(target_ix_combined == 1)[0].shape[0] / target_ix_combined.shape[0]
    #     self_perc_combined = torch.where(target_ix_combined == 2)[0].shape[0] / target_ix_combined.shape[0]
    #     self_rob_combined = torch.where(target_ix_combined == 3)[0].shape[0] / target_ix_combined.shape[0]
    #     self_nonrob_combined = torch.where(target_ix_combined == 4)[0].shape[0] / target_ix_combined.shape[0]
    #
    #
    #
    #     #---------------------
    #     min_all_rob_trn = torch.vstack(
    #         [min_dist_train, min_dist_nonrobust,min_dist_self])
    #
    #     target_ix_rob_trn = torch.argmin(min_all_rob_trn, 0)
    #
    #     train_perc_rob_trn = torch.where(target_ix_rob_trn == 0)[0].shape[0] / target_ix_rob_trn.shape[0]
    #     nonrobust_perc_rob_trn = torch.where(target_ix_rob_trn == 1)[0].shape[0] / target_ix_rob_trn.shape[0]
    #     self_perc_rob_trn = torch.where(target_ix_rob_trn == 2)[0].shape[0] / target_ix_rob_trn.shape[0]


    if prev_training:
        augmenter.set_train()

    if compare_robust:
        results = {
            'lpips_train': train_perc,
            'lpips_test': test_perc,
            'lpips_self': self_perc,
            #
            'lpips_nonrobust_robexp': nonrobust_perc_robexp,
            'lpips_train_robexp': train_perc_robexp,
            'lpips_self_robexp': self_perc_robexp,
            #
            # 'lpips_train_combined': train_perc_combined,
            # 'lpips_test_combined': test_perc_combined,
            # 'lpips_self_combined': self_perc_combined,
            # 'roblpips_robust_combined': self_rob_combined,
            # 'roblpips_nonrobust_combined': self_nonrob_combined,
            # #
            # 'robtrnlpips_train': train_perc_rob_trn,
            # 'robtrnlpips_nonrobust': nonrobust_perc_rob_trn,
            # 'robtrnlpips_self': self_perc_rob_trn,
        }
    else:
        results = {
            'lpips_train': train_perc,
            'lpips_test': test_perc,
            'lpips_self': self_perc,
        }
    print(results)

    return results


if __name__ == "__main__":
    print('ok')
