import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2D(nn.Module):
    '''
        https://discuss.pytorch.org/t/about-segmentation-loss-function/2906/6
    '''

    def __init__(self, reduction='mean', ignore_label=255):
        super(CrossEntropyLoss2D, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)
        return loss


if __name__ == '__main__':
    import torch
    predict = torch.randn((2, 21, 512, 512))
    gt = torch.randint(1, 20, (2, 512, 512))

    loss_function = CrossEntropyLoss2D()
    result = loss_function(predict, gt)
    print(result)

