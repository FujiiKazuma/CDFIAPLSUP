from torch import nn
import torch


def CELoss(output, target):
    # n = torch.sum(target)
    tmp =  torch.sigmoid(output)

    tmp1 = torch.log(tmp + torch.tensor(1e-30))
    tmp1 = tmp1 * target

    tmp2 = torch.log(1 - tmp + torch.tensor(1e-30))
    tmp2 = tmp2 * (1 - target)

    tmp = tmp1 + tmp2
    if any(tmp == -float("inf")):
        tmp[tmp == -float("inf")] = torch.tensor(-100.)  # 応急処置
    if any(tmp == float("nan")):
        print("stop")
    
    loss = -target*tmp

    return loss

def MSELoss(output, target):
    output = torch.sigmoid(output) + torch.tensor(1e-30)
    tmp = output - 1
    tmp = tmp ** 2
    loss = tmp * target

    return loss

class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss,self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.)
    
    def forward(self, inp, target):
        positive, unlabeled = target == self.positive, target == self.unlabeled  # positiveはPなやつだけ1になってる unlabeledはUなやつだけ1になってる
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if inp.is_cuda:
            self.min_count = self.min_count.cuda()
            self.prior = self.prior.cuda()
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))  # negativeのnでなくnumberのn

        # y_positive = self.loss_func(inp)
        # y_unlabeled = self.loss_func(-inp)
        y_positive = CELoss(inp, positive)
        y_positive_inv = CELoss(-inp, positive)
        y_unlabeled = CELoss(-inp, unlabeled)
        # y_positive = MSELoss(inp, positive)
        # y_positive_inv = MSELoss(-inp, positive)
        # y_unlabeled = MSELoss(-inp, unlabeled)

        # positive_risk = torch.sum(self.prior * positive * y_positive / n_positive)
        # negative_risk = torch.sum(-self.prior * positive * y_unlabeled / n_positive + unlabeled * y_unlabeled / n_unlabeled)
        positive_risk = self.prior * torch.sum(y_positive)/ n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv)/ n_positive + torch.sum(y_unlabeled)/n_unlabeled
        negative_risk = 0 if negative_risk < 0 else negative_risk


        if negative_risk < -self.beta and self.nnPU:
            # return -self.gamma * negative_risk
            return positive_risk - torch.tensor(self.beta)
        else:
            return positive_risk + negative_risk
       