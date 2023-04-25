# Rethinking Data Augmentation for Adversarial Robustness



## How to create an augmentation config
Each config contains a dictionary with 3 main keys: preprocessing,transform, and fwd-pass. 
The format is stored as a dictionary in a yaml
Example:
```YAML
{
  'preprocessing':{
    'resize':{
      'size':32
    },
    'ccrop':{
      'size':32
    },
  },
  'transform':{
    'rcrop':{
      'p':0.5,
      'params':{
        'size':32, 'padding':4
      }
    },
    'hflip':{
      'p':0.5,
      'params':{}
    },
    'cjitter':{
      'p':0.5,
      'params':{
        'brightness':.25, 'contrast':.25, 'saturation':.25, 'hue':0
      }
    },
    'rrot':{
      'p':0.5,
      'params':{
        'degrees':2
      }
    },
  },
  'fwd-pass':{
    'mixup':{
      'p':0.5,
      'params':{
        'alpha':1.
      }
    },
  }
}

```

#### 1. preprocessing
It's for preprocessing the input and is limited to resize and centercrop. 
```yaml
  'preprocessing':{
    'resize':{
      'size':32
    },
    'ccrop':{
      'size':32
    },
  },
```
#### 2. transform
It sets the customized torchvision transforms to the dataset, and applies such transform-based augmentations with a given probability. 
The currently supported transforms are: random crop, horizontal flipping, colour jotter, random rotation.
```yaml
 'transform':{
    'rcrop':{
      'p':0.5,
      'params':{
        'size':32, 'padding':4
      }
    },
    'hflip':{
      'p':0.5,
      'params':{}
    },
    'cjitter':{
      'p':0.5,
      'params':{
        'brightness':.25, 'contrast':.25, 'saturation':.25, 'hue':0
      }
    },
    'rrot':{
      'p':0.5,
      'params':{
        'degrees':2
      }
    },
  },
```

#### 3. fwd-pass
These are augmentations that are applied in the forward pass (not using the transform in the dataset class).

Current supported methods are: mixup, adversarial training, and transform-based augmentations (random crop, horizontal flipping, colour jotter, random rotation) known as fwd-trans.
The difference between `fwdtrans` in `fwd-pass`, with `transform` explained above is that:
1.  `transform` is applied in the dataset class, and using original data.
2. `fwd-trans` is applied in the forward pass, and can be applied on already augmented data such as mixedup samples or adversarial examples.
3. in `transform` the augmentation is applied with probability `p` on each sample, while in `fwd-trans` the augmentation is applied with probability `p` on each minibatch.
meaning in `transform` there could be a mixture of augmented and non-augmented samples, while in `fwd-trans` a minibatch is either fully augmented or not augmented by `fwd-trans` augmentations at all. 


```yaml
 'fwd-pass':{
    'adv':{
      'p':0.5,
      'params':{
        'eps_min':0.25,
        'eps_max':0.5,
        'relative_step_size':0.2,
        'constraints':['2'],
        'iterations_min':1,
        'iterations_max':5,
        'targeted':0
      }
    },
    'fwdtrans':{
      'apply_if':'adv',
      'params':{
        'rcrop':{
          'size':32, 'padding':4
        },
        'hflip':{},
        'cjitter':{
          'brightness':.25, 'contrast':.25, 'saturation':.25, 'hue':0
        },
        'rrot':{
          'degrees':2
        },
      }
    },
    'mixup':{
      'apply_if':'adv',
      'params':{
        'alpha':1.
      }
    },
  }
```

The order in which `fwd-pass` augmentations are given in the config defines the order the data is augmented.
In the following example it is adv ->  fwdtrans -> mixup. 
Meaning first adversarial examples are created, then fwdtrans is applied on them, then the resulting samples are mixed up.

There are two ways to control randomness in `fwd-pass`: 1) set `p` with a desired probability of augmentation and 2) use  `apply_if`:**target_aug** to be applied if 'target_aug' was applied.
In the former, we set a specific probability in which that augmentation will be applied. 
For example, in the following, `mixup` and `adv` will be independently applied each with the probability of 0.5:
```yaml
'mixup':{
      'p':0.5,
      'params':{
        'alpha':1.
      }
    },
'adv':{
      'p':0.5,
      'params':{
        'eps_min':0.25,
        'eps_max':0.5,
        'relative_step_size':0.2,
        'constraints':['2'],
        'iterations_min':1,
        'iterations_max':5,
        'targeted':0
      }
    },
```
In the latter, an augmentation is applied only the **target_aug** is applied on a minibatch.
For example, `adv` is applied in minibatches that are augmented with `mixup`:
```yaml
'mixup':{
      'p':0.5,
      'params':{
        'alpha':1.
      }
    },
'adv':{
     'apply_if':'adv',
      'params':{
        'eps_min':0.25,
        'eps_max':0.5,
        'relative_step_size':0.2,
        'constraints':['2'],
        'iterations_min':1,
        'iterations_max':5,
        'targeted':0
      }
    },
```

### Augmentation Config Usage:
For each experiment, a separate augmentation config has to be created and saved and properly named. E.g, `cls_0.5.yaml` could refer to an augmentation using transofm augmentations (also known as classic) with probability of 0.5.
If `job_type` was not set, the name of the augmentation config will be used as job_type for aggregating plots in W&B. 


### Config file naming rules
We use arrows (`->`,`<->`) to denote if two augmentations are applied dependent on another (`->'), or independently from eachother (`<->`).
Example: apply `cls_0.5` independently, and only apply `fwdcls` on samples augmented with `l2adv_0.5`: 
```yaml
cls_0.5<->l2adv_0.5->fwdcls.yaml
```
and here is how such a config looks like:
```yaml
{
  'preprocessing':{
    'resize':{
      'size':32
    },
    'ccrop':{
      'size':32
    },
  },
  'transform':{
    'rcrop':{
      'p':0.5,
      'params':{
        'size':32, 'padding':4
      }
    },
    'hflip':{
      'p':0.5,
      'params':{}
    },
    'cjitter':{
      'p':0.5,
      'params':{
        'brightness':.25, 'contrast':.25, 'saturation':.25, 'hue':0
      }
    },
    'rrot':{
      'p':0.5,
      'params':{
        'degrees':2
      }
    },
  },
  'fwd-pass':{
    'adv':{
      'p':0.5,
      'params':{
        'eps_min':0.25,
        'eps_max':0.5,
        'relative_step_size':0.2,
        'constraints':['2'],
        'iterations_min':1,
        'iterations_max':5,
        'targeted':0
      }
    },
    'fwdtrans':{
      'apply_if':'adv',
      'params':{
        'rcrop':{
          'size':32, 'padding':4
        },
        'hflip':{},
        'cjitter':{
          'brightness':.25, 'contrast':.25, 'saturation':.25, 'hue':0
        },
        'rrot':{
          'degrees':2
        },
      }
    },
  }
}

```


## Logging with W&B



### Install
```bash
cd install
conda env create -f environment.yml
conda activate dar
```
### Set Path

In `path_info/db.path`, set the path for datasets you want to use as follows. Most of datasets will be downloaded (except for ImageNet and CIFAR10C):
```text
cifar10:/{your_path}/data/cifar10
cifar10c:/{your_path}/data/CIFAR10_C
cifar100:/{your_path}/data/cifar100
svhn:/{your_path}/data/svhn
mnist:/{your_path}/data/mnist
mnistc:/{your_path}/data/mnist_c
fmnist:/{your_path}/data/fmnist
fmnistc:/{your_path}/data/FMNIST_C
kmnist:/{your_path}/data/kmnist
imagenet:/{your_path}/data/imagenet
models:/{your_path}/models/DAR_Journal
```

### Create a W&B profile
Since we will be logging experiment in [W&B](wandb.ai), every user that wants to run experiments/work with the library with the logging options activated, requires an account on W&B. 

Use your key in your profile and install the library and log into your account as follows: 

```bash
pip install --upgrade wandb
wandb login your_key
```

### Training
start training a model for a given augmentation config (cls), then applies both targeted and nontargeted PGD attacks:
```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train -n-workers 4 -augmentation-config-file "/path/to/augmentation_dicts/cls.yaml"  -job-type cls_0.5 -training-seed 44864
```
note that job-type here keeps the kind and amount of augmentation. It can be used to group the plots across different seeds in the W&B web UI.



### Generating augmented data
In order to generate some augmented data given a config, you can use the following script. It saves the files in a given folder, using the provided config.

```bash
CUDA_VISIBLE_DEVICES=0 python -m generation.generate_augmented_data -n-workers 4 -augmentation-config-file "/path/to/augmentation_dicts/cls.yaml"  -name test_cls_0.5 -num-epoch 1 -save-dir /path/to/test_generation --normalize-augmented-data
```

