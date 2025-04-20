import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions import dirichlet
import numpy as np

def adaad_inner_loss(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv


def adaad_inner_loss_kl_cluster(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            adv_outputs=F.log_softmax(model(x_adv), dim=1)
            t_adv_outputs = F.softmax(teacher_model(x_adv), dim=1)
            loss_kl = criterion_kl(adv_outputs,t_adv_outputs)
            kl_loss1_cluster = nn.KLDivLoss()(F.log_softmax(adv_outputs.T, dim=1),
                                          F.softmax(t_adv_outputs.detach().T, dim=1))
            loss_kl = torch.sum(loss_kl)
            loss_kl_cluster = torch.sum(kl_loss1_cluster)
        loss=loss_kl+loss_kl_cluster
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv

def adaad_inner_loss_featurealign(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv

def Madry_PGD(model, x_ori, y,
              step_size=2/255,
              steps=10,
              epsilon=8/255,
              norm='L_inf',
              BN_eval=True,
              random_init=True,
              clip_min=0.0,
              clip_max=1.0):

    criterion = nn.CrossEntropyLoss()

    if BN_eval:
        model.eval()
    if random_init:
        x_adv = x_ori.detach() + 0.001 * torch.randn(x_ori.shape).cuda().detach()
    else:
        x_adv = x_ori.detach()
    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    if norm == 'L_inf':
        for _ in range(steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_i = criterion(model(x_adv), y)

            grad = torch.autograd.grad(loss_i, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_ori - epsilon), x_ori + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)
    else:
        raise NotImplementedError
    if BN_eval:
        model.train()
    return x_adv

def label_smoothing(teacher_output, epsilon, c, num_classes=10):
    dirich = torch.from_numpy(np.random.dirichlet(np.ones(num_classes), teacher_output.size(0))).cuda()
    sr = (torch.ones(teacher_output.size(0)).cuda() * (c*epsilon)).unsqueeze(1).repeat(1, num_classes)
    ones = torch.ones_like(sr)
    y_tilde = (ones - sr) * teacher_output + sr * dirich
    return y_tilde

def label_smoothing_nodirichlet(teacher_output, epsilon, c, num_classes=10):
    average= torch.full((teacher_output.size(0), 10), 0.1).cuda()
    sr = (torch.ones(teacher_output.size(0)).cuda() * (c*epsilon)).unsqueeze(1).repeat(1, num_classes)
    ones = torch.ones_like(sr)
    y_tilde = (ones - sr) * teacher_output + sr * average
    return y_tilde

def adaad_inner_loss_cat_nols(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilons=None,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0,
                     c=10):
    # define KL-loss
    epsilons_teacher=epsilons.clone().detach()
    epsilons = epsilons[:,None,None,None].repeat(1, x_natural.size(1), x_natural.size(2), x_natural.size(3))
    criterion_kl = nn.KLDivLoss(reduction='none')
    
    
    # set eval mode formodel
    model.eval()
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))#试试四种方案，一种是softmax后进label_smooth，一种是label_smooth后softmax，或者在softmax后进label_smooth再做一次softmax
            loss_kl = torch.sum(loss_kl)                                  #或者在这里干脆不加
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilons), x_natural + epsilons)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv

def adaad_inner_loss_cat_labelsmooth_before_softmax(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilons=None,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0,
                     c=10):
    # define KL-loss
    epsilons_teacher=epsilons.clone().detach()
    epsilons = epsilons[:,None,None,None].repeat(1, x_natural.size(1), x_natural.size(2), x_natural.size(3))
    criterion_kl = nn.KLDivLoss(reduction='none')
    
    
    # set eval mode formodel
    model.eval()
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(label_smoothing(teacher_model(x_adv),epsilons_teacher,c=c), dim=1))#试试四种方案，一种是softmax后进label_smooth，一种是label_smooth后softmax，或者在softmax后进label_smooth再做一次softmax
            loss_kl = torch.sum(loss_kl)                                  #或者在这里干脆不加
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilons), x_natural + epsilons)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv

def adaad_inner_loss_cat_nolabelsmooth(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilons=None,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0,
                     c=10):
    # define KL-loss
    epsilons = epsilons[:,None,None,None].repeat(1, x_natural.size(1), x_natural.size(2), x_natural.size(3))
    criterion_kl = nn.KLDivLoss(reduction='none')
    # set eval mode formodel
    model.eval()
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))#试试四种方案，一种是softmax后进label_smooth，一种是label_smooth后softmax，或者在softmax后进label_smooth再做一次softmax
            loss_kl = torch.sum(loss_kl)                                  #或者在这里干脆不加
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilons), x_natural + epsilons)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv

def ard_cat_loss(model,
                     x_natural,
                     targets,
                     step_size=2/255,
                     steps=10,
                     epsilons=None,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    epsilons = epsilons[:,None,None,None].repeat(1, x_natural.size(1), x_natural.size(2), x_natural.size(3))
    criterion_kl = nn.KLDivLoss(reduction='none')
    # set eval mode formodel
    model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), targets, size_average=False)                                  #或者在这里干脆不加
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilons), x_natural + epsilons)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv



def ard_ori_loss(net,inputs,targets,step_size=2/255,steps=10,epsilon=8/255):
    x = inputs.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    for i in range(steps):
        x.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(net(x), targets, size_average=False)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size*torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
        x = torch.clamp(x, 0.0, 1.0)
    return x

