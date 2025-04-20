'''Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.
    - set_lr : set the learning rate
    - clip_gradient : clip gradient
'''

import os
import sys
import time
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
# term_width = 40

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.

#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')

#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time

#     L = []
#     if msg:
#         L.append(' | ' + msg)

#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')

#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))

#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

## Input interpolation functions
def mix_data(x, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(2, 2)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, index, lam

def lr_decay(epoch,args):
    if args.lr_schedule == 'piecewise':
        if args.epochs == 200:
            epoch_point = [100,150]
        elif args.epochs == 110: 
            epoch_point = [100, 105] # Early stop for Madry adversarial training
        else:
            raise ValueError
        if epoch < epoch_point[0]:
            if args.warmup_lr and epoch < args.warmup_lr_epoch:
                return 0.001 + epoch / args.warmup_lr_epoch * (args.lr_max-0.001)
            return args.lr_max
        if epoch < epoch_point[1]:

            return args.lr_max / 10
        else:
            return args.lr_max / 100
    elif args.lr_schedule == 'cosine':
        if args.warmup_lr:
            if epoch < args.warmup_lr_epoch:
                return 0.001 + epoch / args.warmup_lr_epoch * (args.lr_max-0.001)
            else:
                return np.max([args.lr_max * 0.5 * (1 + np.cos((epoch-args.warmup_lr_epoch) / (args.epochs-args.warmup_lr_epoch) * np.pi)), 1e-4])
        return np.max([args.lr_max * 0.5 * (1 + np.cos(epoch / args.epochs * np.pi)), 1e-4])
    elif args.lr_schedule == 'constant':
        return args.lr_max
    else:
        raise NotImplementedError
    
class MIXSTDLoss(nn.Module):
    def __init__(self,opt):
        super(MIXSTDLoss, self).__init__()
        self.opt = opt
        self.cross_ent = nn.CrossEntropyLoss()
        self.KL = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, logit_s, logit_t, target):

        stdt = torch.std(logit_t, dim=-1,keepdim=True)
        stds = torch.std(logit_s, dim=-1, keepdim=True)

        ## CLS ##        
        loss = -F.log_softmax(logit_s/stds,-1) * target
        loss_cls = self.opt.gamma * (torch.sum(loss))/logit_s.shape[0]        
        ## STD KD ## 
        p_s = F.log_softmax(logit_s/stds, dim=1)
        p_t = F.softmax(logit_t/stdt, dim=1)
        std_KD = self.KL(p_s, p_t) 
        loss_div = self.opt.alpha * std_KD

        return loss_cls+loss_div

## Partial Mixup
def PMU(inputs, targets, percent=0.9, beta_a=1.0, mixup=True):
    batch_size = inputs.shape[0]

    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, 100).cuda()
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, 100).cuda()
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = np.random.beta(beta_a, beta_a, [batch_size, 1])
    else:
        a = np.ones((batch_size, 1))
    
    ridx = torch.randint(0,batch_size,(int(a.shape[0] * percent),))
    a[ridx] = 1.

    b = np.tile(a[..., None, None], [1, 3, 32, 32])

    inputs1 = inputs1 * torch.from_numpy(b).float().cuda()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float().cuda()

    c = np.tile(a, [1, 100])

    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float().cuda()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float().cuda()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle


## Full Mixup
def FMU(inputs, targets, beta_a=1.0, mixup=True):
    batch_size = inputs.shape[0]

    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, 100)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, 100)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = np.random.beta(beta_a, beta_a, [batch_size, 1])
    else:
        a = np.ones((batch_size, 1))

    b = np.tile(a[..., None, None], [1, 3, 32, 32])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    c = np.tile(a, [1, 100])

    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle

def adjust_weight(epoch,min_weight_clean=0.5,base_weight_clean=1.0,beta=5,alpha=-3,warmup_epoch=60):
    weight=min_weight_clean+(base_weight_clean-min_weight_clean)/2.0*(1-math.erf((beta-alpha)*epoch/warmup_epoch+alpha))
    return weight


def criterion_crd(logits,i):
    loss_crd_i=torch.tensor(0.).cuda()
    
    for j in range(0,i):
        loss_crd_i+=F.kl_div(F.softmax(logits[j],dim=1).log(),F.softmax(logits[i],dim=1),reduction="batchmean")
    return loss_crd_i


class Cat_dataloader(Dataset):
    def __init__(self, data, is_train=True, transform=None):
        self.data = datasets.__dict__[data.upper()]('/data/lsb/dataset/cifar100', train=is_train, download=True, transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        image = self.data[i][0]
        label = self.data[i][1]
        index = i
        return image, label, index
    
def hellinger_distance(p, q):
    # 计算每个分布的概率密度函数的平方根
    sqrt_p = torch.sqrt(p)
    sqrt_q = torch.sqrt(q)
    
    # 计算平方根密度函数之间的差异
    diff = sqrt_p - sqrt_q
    
    # 计算差异的平均值
    mean_diff = torch.mean(diff)
    
    # 对结果取平方根得到 Hellinger 距离
    hellinger = torch.sqrt(torch.abs(mean_diff))
    
    return hellinger