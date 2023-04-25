
import torch
import numpy as np
from losses.base import AugLossCls


class MixupLoss(AugLossCls):
    def __init__(self, criterion, device):
        super().__init__(device)
        self._criterion = criterion
        self.criterion_fn = self._compile_criterion()

    def _compile_criterion(self):
        active_device = self._device
        orig_criterion = self._criterion

        def mixup_criterion_v1(pred, target, args):
            apply_mixup = np.unique(args['mxup']['apply'])
            assert len(apply_mixup) == 1
            if not apply_mixup[0]:
                return orig_criterion(pred, target)
            lam, index = args['mxup']['lam'], args['mxup']['index']
            assert len(np.unique(lam)) == 1
            assert target.shape[0] == len(index)
            try:
                assert max(index) == target.shape[0] - 1
            except:
                print('index mismatch {} / {} / {}'.format(max(index), target.shape[0], np.unique(index), ))
            assert min(index) == 0
            lam = np.unique(lam)[0]
            target_a, target_b = target, target[index]
            if not torch.is_tensor(target_a):
                target_a = torch.tensor(target_a)  # , dtype=torch.float32)
            if not torch.is_tensor(target_b):
                target_b = torch.tensor(target_b)  # , dtype=torch.float32)
            target_a, target_b = target_a.to(active_device), target_b.to(active_device)
            return lam * orig_criterion(pred, target_a) + (1 - lam) * orig_criterion(pred, target_b)

        def mixup_criterion_v2(mixed_pred, target_a, target_b, args):
            lam = args['mxup']['lam']
            lam = np.unique(lam)[0]

            if not torch.is_tensor(target_a):
                target_a = torch.tensor(target_a)  # , dtype=torch.float32)
            if not torch.is_tensor(target_b):
                target_b = torch.tensor(target_b)  # , dtype=torch.float32)
            target_a, target_b = target_a.to(active_device), target_b.to(active_device)
            return lam * orig_criterion(mixed_pred, target_a) + (1 - lam) * orig_criterion(mixed_pred, target_b)

        return mixup_criterion_v1
