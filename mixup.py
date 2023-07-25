import torch
import numpy as np

def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size_m = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size_m).cuda()
    else:
        index = torch.randperm(batch_size_m)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# tfeatures： 目标域特征
# toutputs: 目标域分布
# ppred : 伪标签

use_cuda = torch.cuda.is_available()

tfeatures_m, ppred_a, ppred_b, lam = self.mixup_data(tfeatures, ppred, self.alpha, use_cuda)
tfeatures_m, ppred_a, ppred_b = map(Variable, (tfeatures, ppred_a, ppred_b))

loss_ce_ad = 0.3 * (lam * loss_gen_ce(toutputs, ppred_a) + (1 - lam) * loss_gen_ce(toutputs, ppred_b))