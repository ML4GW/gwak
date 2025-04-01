import math
from torch.optim.lr_scheduler import LRScheduler

class WarmupCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, total_steps, lr, min_lr, warmup_fraction, last_epoch=-1):
        self.total_steps = total_steps
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = int(total_steps * warmup_fraction)
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        #print(f"Current Step = {current_step}")
        if current_step < self.warmup_steps:
            # Linear warmup
            warmup_lr = self.lr * (current_step / self.warmup_steps)
            #print(f"Linear Stage, lr = {warmup_lr}")
            return [warmup_lr for _ in self.base_lrs]
        else:
            # Cosine annealing
            cosine_steps = current_step - self.warmup_steps
            cosine_total_steps = self.total_steps - self.warmup_steps
            cosine_lr = self.min_lr + (self.lr - self.min_lr) * (1 + math.cos(math.pi * cosine_steps / cosine_total_steps)) / 2
            #print(f"Cosine Stage, lr = {cosine_lr}")
            return [cosine_lr for _ in self.base_lrs]
