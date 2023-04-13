'''
Implementation of commonly used distribution distance measurements, including:
- cosine
- kl
- js
- coral
- mmd
- adv
- mine
- pairwise_dist
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


def cosine(source, target):
    '''Cosine similarity loss'''
    # source, target = source.mean(0), target.mean(0)
    cos = nn.CosineSimilarity(dim=0)
    loss = cos(source, target)
    return loss.mean()


def kl(source, target):
    '''Kullback-Leibler divergence loss'''
    if len(source) < len(target):
        target = target[:len(source)]
    elif len(source) > len(target):
        source = source[:len(target)]
    criterion = nn.KLDivLoss(reduction='batchmean')
    loss = criterion(source.log(), target)
    return loss


def js(source, target):
    '''Jensen-Shannon Divergence loss'''
    if len(source) < len(target):
        target = target[:len(source)]
    elif len(source) > len(target):
        source = source[:len(target)]
    M = .5 * (source + target)
    loss_1, loss_2 = kl_div(source, M), kl_div(target, M)
    return .5 * (loss_1 + loss_2)


def coral(source, target):
    '''Deep Correlation Alignment loss'''
    d = source.size(1)  # feature dimension
    ns, nt = source.size(0), target.size(0)     # number of samples

    # source covariance
    tmp_s = torch.ones((1, ns)).cuda() @ source     # torch.Size: ([1,d])
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / \
        (ns - 1)    # torch.Size: ([d,d])

    # target covariance
    tmp_t = torch.ones((1, nt)).cuda() @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum()
    loss = loss / (4 * d * d)

    return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])   # n_s + n_t
        total = torch.cat([source, target], dim=0)  # (n_samples, d)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))     # (1, n_samples, d) -> (n_samples, n_samples, d)
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))     # (n_samples, 1, d) -> (n_samples, n_samples, d)
        # (n_samples, n_samples)    (x_i - x_j)**2
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]        # len() = n_samples
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]     # len() = n_samples  kernel_val[i].shape:(n_samples, n_samples)
        return sum(kernel_val)      # (n_samples, n_samples)

    def linear_mmd(self, X, Y):
        delta = X.mean(axis=0) - Y.mean(axis=0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])  # n_s
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            return loss


def mmd(source, target, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
    '''Maximum Mean Discrepancy loss

    kernel_type='linear' or 'rbf
    '''
    model = MMD_loss(kernel_type, kernel_mul, kernel_num)
    return model(source, target)


class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def adv(source, target, input_dim=256, hidden_dim=512):
    '''Domain Adversarial Discrepancy loss'''
    domain_loss = nn.BCELoss()
    adv_net = Discriminator(input_dim, hidden_dim).cuda()
    # labels of source domain 1
    domain_src = torch.ones((len(source), 1)).cuda()
    # labels of target domain 0
    domain_tar = torch.zeros((len(target), 1)).cuda()
    reverse_src = ReverseLayerF.apply(source, 1)        # ???
    reverse_tar = ReverseLayerF.apply(target, 1)
    pred_src = adv_net(reverse_src)
    pred_tar = adv_net(reverse_tar)
    loss_s, loss_t = domain_loss(
        pred_src, domain_src), domain_loss(pred_tar, domain_tar)
    loss = loss_s + loss_t
    return loss


class Mine(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(input_dim, hidden_dim)
        self.fc1_y = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


class Mine_estimator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(Mine_estimator, self).__init__()
        self.mine_model = Mine(input_dim, hidden_dim)

    def forward(self, X, Y):
        Y_shuffle = Y[torch.randperm(len(Y))]
        loss_joint = self.mine_model(X, Y)
        loss_marginal = self.mine_model(X, Y_shuffle)
        ret = torch.mean(loss_joint) - \
            torch.log(torch.mean(torch.exp(loss_marginal)))
        loss = ret if ret == 0 else -ret
        return loss


def mine(source, target, input_dim=2048, hidden_dim=512):
    '''Mine loss'''
    model = Mine_estimator(input_dim, hidden_dim).cuda()
    return model(source, target)


def pairwise_dist(X, Y):
    '''pairwise distance for tensor'''
    n, d = X.shape
    m, _ = Y.shape
    assert d == Y.shape[1]
    a = X.unsqueeze(1).expand(n, m, d)
    b = Y.unsqueeze(0).expand(n, m, d)
    return torch.pow(a - b, 2).sum((0, 1, 2))


def pairwise_dist_np(X, Y):
    '''pairwise distance for ndarray'''
    n, d = X.shape
    m, _ = Y.shape
    assert d == Y.shape[1]
    a = np.expand_dims(X, 1)
    b = np.expand_dims(Y, 0)
    a = np.tile(a, (1, m, 1))
    b = np.tile(b, (n, 1, 1))
    return np.power(a - b, 2).sum((0, 1, 2))


def pa_np(X, Y):
    '''pairwise distance for ndarray'''
    XY = np.dot(X, Y.T)
    XX = np.sum(np.square(X), axis=1)
    XX = np.transpose([XX])
    YY = np.sum(np.square(Y), axis=1)
    dist = XX + YY - 2 * XY

    return dist.sum()
