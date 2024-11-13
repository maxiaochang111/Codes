import torch
import torch.nn.functional as F
import numpy as np
# from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def notarget_fgsm_attack(GRL_Net, obs, action, epsilon=0.1):
    feature = obs[0]
    X_in = torch.as_tensor(feature, dtype=torch.float32, device=device)
    ori_X_in = X_in

    # X_in = Variable(X_in, requires_grad=True)
    X_in.requires_grad = True
    now_obs = (X_in, obs[1], obs[2])
    output = GRL_Net(now_obs)
    
    GRL_Net.zero_grad()
    loss = F.cross_entropy(output, action)
    loss.backward()

    adv_X_in = X_in + epsilon*X_in.grad.sign()

    eta = torch.clamp(adv_X_in - ori_X_in, min=-epsilon,max=epsilon)
    X_in =torch.clamp(ori_X_in+eta,min=0,max=1).detach_()
    X_in = X_in.detach()
    X_in.requires_grad_()
    X_in.retain_grad()

    feature[:, 0:2] = X_in[:, 0:2].detach().numpy()

    obs = (feature, obs[1], obs[2])

    return obs


def target_fgsm_attack(GRL_Net, obs, epsilon=0.3):
    feature = obs[0]
    X_in = torch.as_tensor(feature, dtype=torch.float32, device=device)
    ori_X_in = X_in
    
    target_action = [2] * 40
    target_action = torch.as_tensor(target_action, dtype=torch.long, device=device)
    rl_mask = torch.as_tensor(obs[2], dtype=torch.long, device=device)
    target_action = torch.mul(target_action, rl_mask)

    # X_in = Variable(X_in, requires_grad=True)
    X_in.requires_grad = True
    now_obs = (X_in, obs[1], obs[2])
    output = GRL_Net(now_obs)
    
    GRL_Net.zero_grad()
    loss = - F.cross_entropy(output, target_action)
    loss.backward()

    adv_X_in = X_in + epsilon*X_in.grad.sign()

    eta = torch.clamp(adv_X_in - ori_X_in, min=-epsilon,max=epsilon)
    X_in =torch.clamp(ori_X_in+eta,min=0,max=1).detach_()
    X_in = X_in.detach()
    X_in.requires_grad_()
    X_in.retain_grad()

    feature[:, 0:2] = X_in[:, 0:2].detach().numpy()

    obs = (feature, obs[1], obs[2])

    return obs