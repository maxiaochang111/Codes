import torch
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def notarget_Qfunction_attack(GRL_Net, obs, epsilon=0.3, num_iter=10, alpha=0.02):
    feature = obs[0]
    X_in = torch.as_tensor(feature, dtype=torch.float32, device=device)
    ori_X_in = X_in
    
    # random initial point
    X_random = np.random.uniform(-epsilon, epsilon, feature.shape)
    X_initial = np.clip(feature + X_random, 0, 1.0)

    X_in = torch.as_tensor(X_initial, dtype=torch.float32, device=device)
    # X_in = Variable(X_in, requires_grad=True)
    
    for i in range(num_iter):
        X_in.requires_grad = True
        now_obs = (X_in, obs[1], obs[2])
        GRL_Net.zero_grad()
        Qvalue = GRL_Net(now_obs)
        
        loss = value_loss(Qvalue)
        loss.backward()
        
        adv_X_in = X_in+alpha*X_in.grad.sign()

        eta = torch.clamp(adv_X_in - ori_X_in, min=-epsilon,max=epsilon)
        X_in =torch.clamp(ori_X_in+eta,min=0,max=1).detach_()
        X_in = X_in.detach()
        X_in.requires_grad_()
        X_in.retain_grad()

    feature[:, 0:2] = X_in[:, 0:2].detach().numpy()

    obs = (feature, obs[1], obs[2])

    return obs


def value_loss(Qvalue):
        loss = torch.mean(Qvalue)
        return loss