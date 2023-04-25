
import torch
# from torchvision.datasets.cifar import CIFAR10
# from torchvision.datasets.folder import ImageFolder
from torchvision.datasets import VisionDataset
from PIL import Image

# from transforms.trans_utils import get_all_deterministic_transforms_dict
# from transforms.trans_utils import  get_transforms_dict, get_deterministic_transforms_dict, \
from torchvision import transforms
import os
import numpy as np
from typing import Any, Callable, Optional, Tuple


class CIFAR10CDSET(VisionDataset):
    r"""CIFAR10 dataset with deterministic transform-based augmentations.
    This class supports robustness lib and its attacks.
    The normalisation is applied in the fwd pass of the robustness lib attack class.
    Hence, no normalisation is needed in the transform.
    Please see make_and_restore_model in Robustness lib.
    """

    def __init__(self, root, device, corruption, severity, transform=None, target_transform=None,
                 normalise=False):
        super().__init__(root, transform, target_transform)

        self.normalise = normalise
        self.image_size = 32
        self.n_channel = 3
        self.num_classes = 10
        self.mean = torch.tensor([(0.4914, 0.4822, 0.4465)], device=device)
        self.std = torch.tensor([(0.2023, 0.1994, 0.2010)], device=device)
        self.transform = transform
        self.corruption = corruption

        self.targets = np.uint8(np.load(os.path.join(root, str(severity), corruption + '_labels.npy')))
        self.data  = np.uint8(np.load(os.path.join(root, str(severity), corruption + '.npy'))) * 255
        self.dataset_len = self.targets.__len__()



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # convert to tensor
        # img = transforms.ToTensor()(img)

        # normalise
        if self.normalise:
            img = transforms.Normalize(mean=tuple(self.mean.detach().cpu().numpy().flatten()),
                                       std=tuple(self.std.detach().cpu().numpy().flatten()))(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_model(self, arch, pretrained):
        """
        This part is added to support Robustness lib attacks.
        :param arch:
        :param pretrained:
        :return:

        Example (with robustness):
        >>> from robustness.model_utils import make_and_restore_model
        >>> net, _ = make_and_restore_model(arch='resnet18', dataset=train_set)
        >>> outputs, _ = net(X, make_adv=False)

        Example (without robustness):
        >>> from architectures.cifar_models import resnetmixup18
        >>> net = resnetmixup18()
        >>> outputs = net(X)
        """
        from architectures import cifar_models
        if pretrained:
            raise ValueError('CIFAR10 does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

    def __len__(self):
        return self.dataset_len
