import torch


def normal_kld(mu, logvar, *args, **kwargs):
    kld = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def gaussian_nll(x, mu, logvar, *args, **kwargs):
    loss = torch.mean((mu - x).pow(2) / (logvar.exp() * 2) + 0.5 * logvar)
    return loss


def l1_reg(model):
    l1_reg = 0
    for weight in model.parameters():
        l1_reg += weight.norm(1)
    return l1_reg


def l2_reg(model):
    l2_reg = 0
    for weight in model.parameters():
        l2_reg += weight.norm(2)
    return l2_reg
