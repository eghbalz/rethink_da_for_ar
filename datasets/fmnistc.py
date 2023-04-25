
import torch
# from torchvision.datasets.mnist import FashionMNIST
from torchvision.datasets import VisionDataset
from PIL import Image

from augment_transforms.trans_utils import get_transforms
from torchvision import transforms
from torchvision import transforms
import os
import numpy as np
from typing import Any, Callable, Optional, Tuple


class FashionMNISTCDSET(VisionDataset):
    r"""FashionMNIST dataset with deterministic transform-based augmentations.
    This class supports robustness lib and its attacks.
    The normalisation is applied in the fwd pass of the robustness lib attack class.
    Hence, no normalisation is needed in the transform.
    Please see make_and_restore_model in Robustness lib.
    """

    def __init__(self, root, device, corruption, severity, transform=None, target_transform=None,
                 normalise=False):
        super().__init__(root, transform, target_transform)

        self.normalise = normalise
        self.image_size = 28
        self.n_channel = 1
        self.num_classes = 10
        self.mean = torch.tensor([0.1307], device=device)
        self.std = torch.tensor([0.3081], device=device)
        self.corruption = corruption
        self.targets = np.uint8(np.load(os.path.join(root, str(severity), corruption + '_labels.npy')))
        self.data = np.uint8(np.load(os.path.join(root, str(severity), corruption + '.npy'))) * 255
        self.dataset_len = self.targets.__len__()

    def __getitem__(self, index):
        """
        This method applies the transform augmentations based on the predefined values.
        :param index:
        :return:
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # convert to tensor
        # img = transforms.ToTensor()(img)
        if self.transform is not None:
            img = self.transform(img)
        # normalise
        # if self.normalise:
        #     img = transforms.Normalize(mean=tuple(self.mean.detach().cpu().numpy().flatten()),
        #                                std=tuple(self.std.detach().cpu().numpy().flatten()))(img)

        # transform target if necessary
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
        >>> from architectures.mnist_models import resnetmixup18
        >>> net = resnetmixup18()
        >>> outputs = net(X)
        """
        from architectures import mnist_models
        if pretrained:
            raise ValueError('MNIST does not support pytorch_pretrained=True')
        return mnist_models.__dict__[arch](num_classes=self.num_classes)

    def __len__(self):
        return self.dataset_len
