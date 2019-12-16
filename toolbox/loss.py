import torch.nn as nn
import torch.nn.functional as F


class MscCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(MscCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                        ignore_index=self.ignore_index, reduction=self.reduction)
        return loss / len(input)


if __name__ == '__main__':
    import torch

    predict = torch.randn((2, 21, 512, 512))
    gt = torch.randint(1, 20, (2, 512, 512))

    loss_function = nn.CrossEntropyLoss()
    result = loss_function(predict, gt)
    print(result)
