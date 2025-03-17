# by Ahmadreza Attarpour, a.attarpour@mail.utoronto.ca, Tony Xu, tonyxu74@hotmail.com, Grace Yu gracefengqing.yu@mail.utoronto.ca

from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineScheduler(_LRScheduler):
    """ 
    This code provide the learning rate scheduler for the optimizer.
    It first goes warm up (learning rate increases), and then cosine (learning rate decreases).
    The suggested warm up epochs should be 10% of the overall epochs.
    
    """

    def __init__(self, optimizer, warmup_epochs, max_lr, max_epochs, min_lr=0.0, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        self.period = self.max_epochs - self.warmup_epochs

        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # # inside the warmup epochs
        # if self.last_epoch < self.warmup_epochs:
        #     curr_lr = self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch / self.warmup_epochs
        # # cosine schedule after warmup
        # else:
        #     curr_epochs = self.last_epoch - self.warmup_epochs
        #     curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
        #             1 + math.cos(math.pi * curr_epochs / self.period)
        #     )

        # return [curr_lr for group in self.optimizer.param_groups]

        # inside the warmup epochs
        if self.last_epoch < self.warmup_epochs:
            curr_mult = self.last_epoch / self.warmup_epochs
        # cosine schedule after warmup
        else:
            curr_epochs = self.last_epoch - self.warmup_epochs
            curr_mult = 0.5 * (
                1 + math.cos(math.pi * curr_epochs / self.period)
            )

        return [curr_mult * baselr for baselr in self.base_lrs]

                
    
    