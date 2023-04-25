
import torch
import re
from torch._six import container_abcs, string_classes, int_classes
import numpy as np
from abc import ABC, abstractmethod



class AugCollateCls(object):

    def __init__(self, device):
        self._device = device
        self.collate_fn = self._compile_collate()
        pass

    def _compile_collate(self):
        """This is only to returns the actual collate function. Can rename aug_collate to your desired name."""
        return self.aug_collate

    def aug_collate(self,batch):
        """Implement the collate fucntion. Can rename aug_collate to your desired name."""
        pass

    def _preprocessing(self):
        """Implement the preprocessing function."""
        pass
