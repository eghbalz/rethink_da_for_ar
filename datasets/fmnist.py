
import torch
from torchvision.datasets.mnist import FashionMNIST
from PIL import Image

from augment_transforms.trans_utils import get_transforms, load_augmentation_config
from torchvision import transforms


class FashionMNISTDSET(FashionMNIST):
    r"""FashionMNIST dataset with deterministic transform-based augmentations.
    This class supports robustness lib and its attacks.
    The normalisation is applied in the fwd pass of the robustness lib attack class.
    Hence, no normalisation is needed in the transform.
    Please see make_and_restore_model in Robustness lib.
    """

    def __init__(self, root, train, augmentation_config_file, device, transform=None, target_transform=None,
                 download=True):
        super().__init__(root, train=train, target_transform=target_transform, download=download)

        self.image_size = 28
        self.n_channel = 1
        self.num_classes = 10
        self.mean = torch.tensor([0.1307], device=device)
        self.std = torch.tensor([0.3081], device=device)
        self.augmentation_config_file = augmentation_config_file
        if transform is None:
            self.augmentation_dict = load_augmentation_config(self.augmentation_config_file)
            self.transform = get_transforms(self.augmentation_dict, ds_name='fmnist')
        else:
            self.augmentation_dict = None
            self.transform = transform

    def __getitem__(self, index):
        """
        This method applies the transform augmentations based on the predefined values.
        :param index:
        :return:
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        input_dict = {'img': img,
                      'target': target,
                      'index': index}
        if self.transform is not None:
            input_dict = self.transform(input_dict)
        return input_dict['img'], target

    def reset_augdict_transform(self, augmentation_dict=None, augmentation_config_file=None):
        if augmentation_dict is not None:
            self.augmentation_dict = augmentation_dict
            self.transform = get_transforms(self.augmentation_dict, ds_name='fmnist')
            print('dset transform has been updated!')
        elif augmentation_config_file is not None:
            self.augmentation_dict = load_augmentation_config(augmentation_config_file)
            self.transform = get_transforms(self.augmentation_dict, ds_name='fmnist')
            print('dset transform has been updated!')
        else:
            raise ValueError('either set augmentation_dict or provide augmentation_config_file')

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
        >>> from architectures.mnist_models import resnet18
        >>> net = resnet18()
        >>> outputs = net(X)
        """
        from architectures import mnist_models
        if pretrained:
            raise ValueError('FMNIST does not support pytorch_pretrained=True')
        return mnist_models.__dict__[arch](num_classes=self.num_classes)

    def __len__(self):
        return len(self.data)
