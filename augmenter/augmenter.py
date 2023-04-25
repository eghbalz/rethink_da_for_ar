
import torch
import torch.nn as nn
import torch.optim as optim
from augment_transforms.trans_utils import get_fwdtrans_aug
from robustness.model_utils import make_and_restore_model

import numpy as np
import random

import pickle


class Augmenter(object):
    """
    This class is not multioprocess-safe.
    """

    def __init__(self, dset, device, arch, lr, milestones, lr_gamma, weight_decay, momentum):
        """
        initializer of the Augmenter class.
        :param dset:
        :param device:
        :param arch:
        :param lr:
        :param milestones:
        :param lr_gamma:
        :param weight_decay:
        :param momentum:
        """

        # model and dataset
        self.dset = dset
        self.arch = arch
        self.device = device
        self.create_model()

        # opt
        self.milestones = milestones
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr
        self.set_opt()

        # init dicts
        self.init_dicts()

        # criterion
        self.set_criterion()

        # set required params
        self.set_fwdpass_aug_dict()

        # set fwd transforms
        self.set_fwdpass_transforms()

    # def __deepcopy__(self, memodict={}):

    def create_model(self):
        """
        create model and move to device.
        :return:
        """
        self.model, _ = make_and_restore_model(arch=self.arch, dataset=self.dset)
        self.model.to(self.device)

    def set_model_states(self, states):
        """

        :return:
        """
        self.model.load_state_dict(states)
        self.model.to(self.device)

    def init_dicts(self):
        """
        initialize empty dicts.
        :return:
        """
        self.apply_dict = {}
        self.fwdpass_aug_dict = {}
        self.generated_params = {}

    def set_criterion(self):
        """
        set the criterion.
        :return:
        """
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

    def set_fwdpass_aug_dict(self):
        """
        set the fwd-pass augmentation dict for each method, respectively.
        :return:
        """
        # order of keys is important.
        if 'fwd-pass' in self.dset.augmentation_dict.keys():
            for aug_f in self.dset.augmentation_dict['fwd-pass'].keys():
                eval('self.set_fwdpass_{}_dict'.format(aug_f))()

    def set_fwdpass_transforms(self):
        """
        get and set fwd-pass transform augmentations
        :return:
        """
        self.fwd_transforms = get_fwdtrans_aug(self.fwdpass_aug_dict)

    def fwdpass_aug_dict_builder(self, aug_key):
        """
        set the dict for the fwdpass augmentatuions from the config.
        :param aug_key:
        :return:
        """
        aug_dict = {}
        if 'fwd-pass' in self.dset.augmentation_dict.keys():
            augmentation_dict = self.dset.augmentation_dict['fwd-pass']
            if aug_key in augmentation_dict.keys():
                aug_dict = augmentation_dict[aug_key]

        self.fwdpass_aug_dict[aug_key] = aug_dict

    def set_fwdpass_adv_dict(self):
        """
        set the dict for fwd-pass adversarial augmentation.
        :return:
        """
        self.fwdpass_aug_dict_builder(aug_key='adv')

    def set_fwdpass_mixup_dict(self):
        """
        set the dict for fwd-pass mixup augmentation.
        :return:
        """
        self.fwdpass_aug_dict_builder(aug_key='mixup')

    def set_fwdpass_manmixup_dict(self):
        """
        set the dict for fwd-pass mixup augmentation.
        :return:
        """
        self.fwdpass_aug_dict_builder(aug_key='manmixup')

    def set_fwdpass_cutmix_dict(self):
        """
        set the dict for fwd-pass cutmix augmentation.
        :return:
        """
        self.fwdpass_aug_dict_builder(aug_key='cutmix')

    def set_fwdpass_fwdtrans_dict(self):
        """
        set the dict for fwd-pass transform-based augmentation.
        :return:
        """
        self.fwdpass_aug_dict_builder(aug_key='fwdtrans')

    def set_opt(self):
        """
        set the optimizer and scheduler.
        :return:
        """
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.milestones, gamma=self.lr_gamma
        )

    def throw_dice_adv(self):
        """
        apply adversarial augmentation or not.
        :return:
        """
        self.aug_or_not(aug_key='adv')

    def throw_dice_mixup(self):
        """
        apply mixup or not.
        :return:
        """
        self.aug_or_not(aug_key='mixup')

    def throw_dice_manmixup(self):
        """
        apply mixup or not.
        :return:
        """
        self.aug_or_not(aug_key='manmixup')

    def throw_dice_cutmix(self):
        """
        apply cutmix or not.
        :return:
        """
        self.aug_or_not(aug_key='cutmix')

    def throw_dice_fwdtrans(self):
        """
        apply fwd-pass transform augmentation or not.
        :return:
        """
        self.aug_or_not(aug_key='fwdtrans')

    def throw_dice(self, aug_key):
        """
        decide whether or not to apply an augmentation (depending on the model, either with probability p, or based on the state of another augmentation)
        :param aug_key:
        :return:
        """
        if 'p' in self.fwdpass_aug_dict[aug_key].keys():
            prob = self.fwdpass_aug_dict[aug_key]['p']
            apply = np.random.rand() < prob
        elif 'apply_if' in self.fwdpass_aug_dict[aug_key].keys():
            apply_key = self.fwdpass_aug_dict[aug_key]['apply_if']
            if apply_key in self.apply_dict.keys():
                apply = self.apply_dict[apply_key]
            else:
                assert False
        else:
            assert False

        return apply

    def throw_dice_aug(self):
        """
        calls the respective throw_dice function for the fwdpass augmentations, according to the dict.
        :return:
        """
        for aug_f in self.fwdpass_aug_dict.keys():
            eval('self.throw_dice_{}'.format(aug_f))()

    # ============
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """
        compiles the mixup augmentation criterion.

        :param pred:
        :param y_a:
        :param y_b:
        :param lam:
        :return:
        """
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_criterion(self, pred, y_a, y_b, lam):
        """
        compiles the mixup augmentation criterion.

        :param pred:
        :param y_a:
        :param y_b:
        :param lam:
        :return:
        """
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def aug_or_not(self, aug_key):
        """
        apply an augmentation based on a given key, or not
        :param aug_key:
        :return:
        """
        assert aug_key in self.fwdpass_aug_dict.keys()
        apply_aug = self.throw_dice(aug_key)
        self.apply_dict[aug_key] = apply_aug

    def generate_attack(self):
        """
        generate augmentations/their required params.
        :return: Returns mixed inputs, pairs of targets, and lambda
        """
        constraint = random.choice(self.fwdpass_aug_dict['adv']['params']['constraints'])
        attack_epsilon = np.random.uniform(self.fwdpass_aug_dict['adv']['params']['eps_min'],
                                           self.fwdpass_aug_dict['adv']['params']['eps_max'])
        attack_alpha = attack_epsilon * self.fwdpass_aug_dict['adv']['params']['relative_step_size']
        attack_iters = np.random.randint(self.fwdpass_aug_dict['adv']['params']['iterations_min'],
                                         self.fwdpass_aug_dict['adv']['params']['iterations_max'] + 1)
        targeted = bool(self.fwdpass_aug_dict['adv']['params']['targeted'])
        attack_kwargs = {
            'constraint': constraint,  # use L2-PGD
            'eps': attack_epsilon,  # L2 radius around original image
            'step_size': attack_alpha,
            'targeted': targeted,  # Targeted attack
            'iterations': attack_iters,
            'do_tqdm': False,
        }
        return attack_kwargs

    def generate_mixup(self, x, y):
        """
        generates params and mixup samples.
        :param x:
        :param y:
        :return: Returns mixed inputs, pairs of targets, and lambda
        """

        if self.fwdpass_aug_dict['mixup']['params']['alpha'] > 0:
            lam = np.random.beta(self.fwdpass_aug_dict['mixup']['params']['alpha'],
                                 self.fwdpass_aug_dict['mixup']['params']['alpha'])
        else:
            lam = 1.

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x, y_a, y_b = self.mixup_samples(x, y, index, lam)
        return mixed_x, y_a, y_b, lam

    def generate_manmixup(self, x, y):
        """
        generates params and mixup samples.
        :param x:
        :param y:
        :return: Returns mixed inputs, pairs of targets, and lambda
        """

        if self.fwdpass_aug_dict['manmixup']['params']['alpha'] > 0:
            lam = np.random.beta(self.fwdpass_aug_dict['manmixup']['params']['alpha'],
                                 self.fwdpass_aug_dict['manmixup']['params']['alpha'])
        else:
            lam = 1.

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        # in manifold mixup, we only shuffle lalebs according to generated indexes, and actual mixup operation is done within the network on embeddings.
        y_a, y_b = self.shuffle_labels(y, index)
        layer_id = np.random.choice(self.fwdpass_aug_dict['manmixup']['params']['layer_ids'], 1, replace=False)[0]
        return x, y_a, y_b, lam, index, layer_id

    def generate_cutmix(self, x, y):
        """
        generates params and mixup samples.
        :param x:
        :param y:
        :return: Returns mixed inputs, pairs of targets, and lambda
        """

        if self.fwdpass_aug_dict['cutmix']['params']['alpha'] > 0:
            lam = np.random.beta(self.fwdpass_aug_dict['cutmix']['params']['alpha'],
                                 self.fwdpass_aug_dict['cutmix']['params']['alpha'])
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x, y_a, y_b = self.cutmix_samples(x, y, index, lam)
        return mixed_x, y_a, y_b, lam

    def mixup_samples(self, x, y, index, lam):
        """
        mixup samples, given parameters.
        :param x:
        :param y:
        :param index:
        :param lam:
        :return:
        """
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b

    def shuffle_labels(self, y, index):
        """
        mixup samples, given parameters.
        :param x:
        :param y:
        :param index:
        :param lam:
        :return:
        """
        y_a, y_b = y, y[index]
        return y_a, y_b

    def cutmix_samples(self, x, y, index, lam):
        """
        mixup samples, given parameters.
        :param x:
        :param y:
        :param index:
        :param lam:
        :return:
        """
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        y_a, y_b = y, y[index]
        return x, y_a, y_b

    def generate_fwdtrans(self, X):
        """
        generate fwd-pass transform augmentations.
        :param X:
        :return:
        """
        X_aug = self.fwd_transforms(X)
        return X_aug

    def adv_augment(self, X, y):
        """
        apply adversarial augmentations if flag was set.
        :param X:
        :param y:
        :return:
        """
        if self.apply_dict['adv']:
            attack_kwargs = self.generate_attack()
            X = self.adv_example(X, y, attack_kwargs)
        return X

    def mixup_augment(self, X, y):
        """
        apply mixup augmentations if flag was set.
        :param X:
        :param y:
        :return:
        """
        if 'mixup' in self.apply_dict.keys() and self.apply_dict['mixup']:
            X, y_a, y_b, lam = self.generate_mixup(X, y)
            self.generated_params['mixup'] = [y_a, y_b, lam]
        return X

    def manmixup_augment(self, X, y):
        """
        apply manmixup augmentations if flag was set.
        :param X:
        :param y:
        :return:
        """
        if 'manmixup' in self.apply_dict.keys() and self.apply_dict['manmixup']:
            X, y_a, y_b, lam, index, layer_id = self.generate_manmixup(X, y)
            self.generated_params['manmixup'] = [y_a, y_b, lam, index, layer_id]
        return X

    def cutmix_augment(self, X, y):
        """
        apply cutmix augmentations if flag was set.
        :param X:
        :param y:
        :return:
        """
        if 'cutmix' in self.apply_dict.keys() and self.apply_dict['cutmix']:
            X, y_a, y_b, lam = self.generate_cutmix(X, y)
            self.generated_params['cutmix'] = [y_a, y_b, lam]
        return X

    def fwdtrans_augment(self, X, y):
        """
        apply fwd-pass transform augmentations if flag was set.
        :param X:
        :param y:
        :return:
        """
        if self.apply_dict['fwdtrans']:
            X = self.generate_fwdtrans(X)
        return X

    def augment_data(self, X, y):
        """
        loop over all augmentations and apply if flag was set.
        :param X:
        :param y:
        :return:
        """
        for aug_f in self.fwdpass_aug_dict.keys():
            # X, args = getattr(globals()['self'], '{}_augment'.format(aug_f))(X, y)
            X = eval('self.{}_augment'.format(aug_f))(X, y)

        return X

    def get_loss(self, outputs, y):
        """
        get the appropriate loss.
        :param outputs:
        :param y:
        :return:
        """
        if 'mixup' in self.apply_dict.keys() and self.apply_dict['mixup']:
            y_a, y_b, lam = self.generated_params['mixup']
            loss = self.mixup_criterion(outputs, y_a, y_b, lam)
        elif 'manmixup' in self.apply_dict.keys() and self.apply_dict['manmixup']:
            y_a, y_b, lam, index, layer_id = self.generated_params['manmixup']
            loss = self.mixup_criterion(outputs, y_a, y_b, lam)
        elif 'cutmix' in self.apply_dict.keys() and self.apply_dict['cutmix']:
            y_a, y_b, lam = self.generated_params['cutmix']
            loss = self.cutmix_criterion(outputs, y_a, y_b, lam)
        else:
            loss = self.criterion(outputs, y)
        return loss

    def augment_labels(self, X, y):
        if 'mixup' in self.apply_dict.keys() and self.apply_dict['mixup']:
            y_a, y_b, lam = self.generated_params['mixup']
            y_augmented = lam * y_a + (1 - lam) * y_b
            X_augmented = X
        elif 'manmixup' in self.apply_dict.keys() and self.apply_dict['manmixup']:
            y_a, y_b, lam, index, layer_id = self.generated_params['manmixup']
            y_augmented = lam * y_a + (1 - lam) * y_b
            X_augmented = X
        elif 'cutmix' in self.apply_dict.keys() and self.apply_dict['cutmix']:
            y_a, y_b, lam = self.generated_params['cutmix']
            y_augmented = lam * y_a + (1 - lam) * y_b
            X_augmented = X
        else:
            y_augmented = y
            X_augmented = X

        return X_augmented, y_augmented

    def get_augmented_output(self, X, y):
        """
        throw dice, apply data aug., and predict.
        :param X:
        :param y:
        :return:
        """

        # throw dice
        self.throw_dice_aug()

        # augment data
        X = self.augment_data(X, y)

        # fwd pass
        outputs = self.augmented_predict(X)

        return outputs

    def calc_loss(self, X, y):
        """
        main function responsible for calculating the loss.
        :param X:
        :param y:
        :return:
        """
        self.set_train()

        # throw dice, apply data aug., and predict
        outputs = self.get_augmented_output(X, y)

        # calc loss
        loss = self.get_loss(outputs, y)

        return loss

    def generate_augmented_data(self, X, y):
        """
        main function responsible for generating augmented data
        :param X:
        :param y:
        :return:
        """

        # throw dice
        self.throw_dice_aug()

        # augment data
        X_augmented = self.augment_data(X, y)

        # fwd pass
        X_augmented, y_augmented = self.augment_labels(X_augmented, y)

        return X_augmented.detach(), y_augmented.detach()

    def augmented_predict(self, x):
        """
        predicts on a given batch of samples.
        :param x:
        :return:
        """
        # we need an exception for the predict function of manifild mixup, as the fwd pass needs the generated parameters.
        if 'manmixup' in self.apply_dict.keys() and self.apply_dict['manmixup']:
            outputs, _ = self.model(x, make_adv=False, generated_params=self.generated_params['manmixup'])
        else:
            outputs, _ = self.model(x, make_adv=False)
        return outputs

    def set_eval(self):
        """
        sets model to eval mode.
        :return:
        """
        self.model.eval()

    def set_train(self):
        """
        sets model to train mode.
        :return:
        """
        self.model.train()

    def eval_mode_pred(self, x):
        """
        predicts on a given batch of samples in eval mode.
        :param x:
        :return:
        """
        prev_training = bool(self.model.training)
        self.set_eval()
        with torch.no_grad():
            outputs, _ = self.model(x, make_adv=False)
        if prev_training:
            self.set_train()

        return outputs

    def adv_example(self, x, y, attack_kwargs):
        """
        create adversarial examples given the config.
        :param x:
        :param y:
        :param attack_kwargs:
        :return:
        """
        _, adv_x = self.model(x, y, make_adv=True, with_image=True, **attack_kwargs)
        return adv_x

    def update(self, X, y):
        """
        Update the model and backprop the grads.
        :param X:
        :param y:
        :return:
        """
        # calc loss
        self.loss_value = self.calc_loss(X, y)

        # push update
        self.optimizer.zero_grad()
        self.loss_value.backward()
        self.optimizer.step()

    def get_loss_value(self):
        """
        get the value of loss for logging.
        :return:
        """
        return self.loss_value.detach().cpu().numpy().item()

    def calc_eval_loss(self, X, y):
        """
        calculate the loss on a given data for logging.
        :param X:
        :param y:
        :return:
        """
        outputs = self.augmented_predict(X)
        loss = self.criterion(outputs, y)
        return loss.detach().cpu().numpy().item()

    def calc_eval_loss_from_output(self, outputs, y):
        """
        calculate the loss on output activations for logging.
        :param X:
        :param y:
        :return:
        """
        loss = self.criterion(outputs, y)
        return loss.detach().cpu().numpy().item()

    def lr_step(self):
        """
        step for the lr schedule.
        :return:
        """
        self.scheduler.step()

    def save_state(self, file_path):
        """
        save the state of the model in the given path.
        :param file_path:
        :return:
        """
        torch.save(self.model.state_dict(), file_path)

    def load_state(self, file_path):
        """
        load the state of the model in the given path.
        :param file_path:
        :return:
        """
        state_dict = torch.load(file_path)
        return state_dict

    def load_and_set_state(self, file_path):
        """
        load the state of the model in the given path.
        :param file_path:
        :return:
        """
        self.model.load_state_dict(torch.load(file_path))

    def save_augmenter(self, file_path):
        """
        save the augmenter class instance in the given path.
        :param file_path:
        :return:
        """
        raise NotImplementedError('Save Augmenter is not implemented!')
        # todo: not tested!
        # checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        # torch.save(checkpoint, 'Checkpoint.pth')
        # with open(file_path + '.augmenter', 'wb') as f:
        #     pickle.dump(checkpoint , f)

    def load_augmenter(self, file_path):
        """
         load the augmenter instance from the given path.
        :param file_path:
        :return:
        """
        with open(file_path + '.augmenter', 'rb') as f:
            augmenter = pickle.load(f)

        for prop in augmenter:
            self.prop = prop


if __name__ == "__main__":
    from torchvision import transforms, datasets
    from path_info import PATH_DICT
    from datasets import cifar10

    device = torch.device("cuda:0")
    DS_DATAROOT = PATH_DICT['cifar10']
    conf = '/media/sharpie/1TBlade1/src/github/data_augmentation_robustness/augmentation_dicts/cls-on-adv-then-mixup.yaml'
    test_transforms = transforms.Compose([transforms.ToTensor()])
    train_set = cifar10.CIFAR10DSET(DS_DATAROOT,
                                    True, conf, device)
    augmenter = Augmenter(train_set, device, 'resnet18', 0.001, [10], .5, 0, 0.9)

    augmenter.save_augmenter('here.pth')
    X, y = [1], [2]
    args_dict = {}
    for aug_f in ['mixup', 'fwdtrans', 'adv']:
        X, args = getattr(globals()['Augmenter'], '{}_augment'.format(aug_f))(X, y)
        args_dict[aug_f] = args
