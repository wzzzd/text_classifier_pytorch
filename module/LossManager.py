
from torch.nn import CrossEntropyLoss
from module.loss.focal_loss import FocalLoss


class LossManager(object):
    
    def __init__(self, loss_type):
        # 判断配置的loss类型
        if loss_type == 'ce':
            self.loss_func = CrossEntropyLoss()
        elif loss_type == 'focalloss':
            self.loss_func = FocalLoss()
        else:
            self.loss_func = CrossEntropyLoss()
    
    
    def compute(self, input, target):
        """        
        计算loss
        Args:
            input: [N, C]
            target: [N, ]
        """
        loss = self.loss_func(input, target)
        return loss

