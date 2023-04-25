
import torch
import re
from torch._six import container_abcs, string_classes, int_classes
import numpy as np
from collates.base import AugCollateCls

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "mixup_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class DeterministicCollate(AugCollateCls):
    def __init__(self, device):
        super().__init__(device)
        self.collate_fn = self._compile_collate()

    def _compile_collate(self):
        return self.deterministic_collate

    def deterministic_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        # inputs, targets_a, targets_b, lam = mixup_obj.mixup_data(batch[0], batch[1])
        elem = batch[0]
        elem_type = type(elem)
        # goes in for images/data (should be a tensor by now)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        # goes in for labels
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return self.mixup_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):  # for labels (if int) and indexes
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):  # dicts
            return {key: [d[key] for d in batch] for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(samples for samples in zip(*batch)))
        elif isinstance(elem, type(None)): #this handles mx batch which is None
            return batch
        elif isinstance(elem, container_abcs.Sequence):  # first time it goes here to separate data, labels, and indexes
            inputs, targets, augs_info_lst = zip(*batch)
            inputs = self.collate_fn(inputs)
            targets = self.collate_fn(targets)

            # this part is for additional input and targets that an augmentation (such as mixup) might need.
            # mx_inputs = self.collate_fn(mx_inputs)
            # mx_targets = self.collate_fn(mx_targets)

            aug_info_dict = {}
            for augs_info in augs_info_lst:
                for aug_key in augs_info.keys():
                    aug_info = augs_info[aug_key]
                    if aug_key not in aug_info_dict.keys():
                        aug_info_dict[aug_key] = {}
                    for k in aug_info.keys():
                        if k in aug_info_dict[aug_key].keys():
                            aug_info_dict[aug_key][k].append(aug_info[k])
                        else:
                            aug_info_dict[aug_key][k] = [aug_info[k]]

            return inputs, targets, aug_info_dict

        raise TypeError(default_collate_err_msg_format.format(elem_type))
