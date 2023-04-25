


class AugLossCls(object):
    """
    Base class for Aug loss.
    """
    def __init__(self, device):
        self._device = device
        pass

    def __compile_criterion(self):
        def aug_criterion(pred, args):
            pass
        return aug_criterion
