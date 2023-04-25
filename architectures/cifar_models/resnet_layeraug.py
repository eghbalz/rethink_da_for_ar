'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l - 1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, input, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(input)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(input)

        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class ResNet_ManMixup(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(ResNet_ManMixup, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale * widths[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def mixup_latent(self, input, generated_params, current_layer_id):
        y_a, y_b, lam, index, layer_id = generated_params
        if layer_id == current_layer_id:
            input = lam * input + (1 - lam) * input[index]
        return input

    def check_batch(self, args):
        for k in args.keys():
            if 'batch_augment' in args[k].keys():
                try:
                    # only batch_augments need to have the same values in a batch
                    assert len(np.unique(args[k]['batch_augment'])) == 1
                    if np.unique(args[k]['batch_augment'])[0] and 'apply' in args[k].keys():
                        try:
                            assert len(np.unique(args[k]['apply'])) == 1
                        except:
                            print('failed apply unique assert!')
                except:
                    print('failed batch augment unique assert!')
            if 'layer' in args[k].keys():
                try:
                    assert len(np.unique(args[k]['layer'])) == 1
                except:
                    print('aug layers not unique: {}'.format(np.unique(args[k]['layer'])))
            # check if mixup should be applied, them only one lambda value is valid.
            if k == 'mxup':
                try:
                    assert len(np.unique(args[k]['apply'])) == 1
                    # if np.unique(args[k]['apply'])[0]:
                    assert len(np.unique(args[k]['lam'])) == 1
                except:
                    print('something iss not unique!')

    def forward(self, input, **kwargs):
        if 'no_relu' not in kwargs.keys():
            kwargs['no_relu'] = False
        if 'with_latent' not in kwargs.keys():
            kwargs['with_latent'] = False
        if 'fake_relu' not in kwargs.keys():
            kwargs['fake_relu'] = False

        assert (not kwargs['no_relu']), \
            "no_relu not yet supported for this architecture"

        if 'generated_params' in kwargs.keys():
            # for training
            generated_params = kwargs['generated_params']
        else:
            # for inference (no mixup)
            generated_params = None, None, None, None, -1

            # Augmenting layer 0 (inputs)
        input = self.mixup_latent(input, generated_params, current_layer_id=0)

        out = F.relu(self.bn1(self.conv1(input)))
        out = self.layer1(out)

        # Augmenting layer 1
        out = self.mixup_latent(out, generated_params, current_layer_id=1)
        out = self.layer2(out)

        # Augmenting layer 2
        out = self.mixup_latent(out, generated_params, current_layer_id=2)
        out = self.layer3(out)

        # Augmenting layer 3
        out = self.mixup_latent(out, generated_params, current_layer_id=3)
        out = self.layer4(out, fake_relu=kwargs['fake_relu'])

        # Augmenting layer 4
        out = self.mixup_latent(out, generated_params, current_layer_id=4)

        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)

        # Augmenting layer 5
        final = self.mixup_latent(final, generated_params, current_layer_id=5)

        if kwargs['with_latent']:
            return final, pre_out
        return final


def ResNet18_ManMixup(**kwargs):
    return ResNet_ManMixup(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet18Wide_ManMixup(**kwargs):
    return ResNet_ManMixup(BasicBlock, [2, 2, 2, 2], wd=1.5, **kwargs)


def ResNet18Thin_ManMixup(**kwargs):
    return ResNet_ManMixup(BasicBlock, [2, 2, 2, 2], wd=.75, **kwargs)


def ResNet34_ManMixup(**kwargs):
    return ResNet_ManMixup(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50_ManMixup(**kwargs):
    return ResNet_ManMixup(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101_ManMixup(**kwargs):
    return ResNet_ManMixup(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152_ManMixup(**kwargs):
    return ResNet_ManMixup(Bottleneck, [3, 8, 36, 3], **kwargs)


resnet18_manmixup = ResNet18_ManMixup
resnet50_manmixup = ResNet50_ManMixup
resnet101_manmixup = ResNet101_ManMixup
resnet152_manmixup = ResNet152_ManMixup

resnet18thin_manmixup = ResNet18Thin_ManMixup
resnet18wide_manmixup = ResNet18Wide_ManMixup


def test():
    net = resnet18_manmixup()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())