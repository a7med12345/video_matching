import random
import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from collections import namedtuple
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import math
from torch.nn import Parameter
import pickle
from torch import distributed as dist
from torch.utils.data.sampler import Sampler
from torch import nn, autograd, optim
###############################################################################
# Helper Functions & classes
###############################################################################




def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='orthogonal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net,  gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net



def define_G(netG, opt, gpu_ids=[]):
    net = None
    if netG == 'g':
        net = TwinNetwork()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return net
    return init_net(net, gpu_ids)






class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1).pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1).pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()



#############CONV+LSTM


class ConvLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            input_channel,
            hidden_channel,
            kernel_size,
            stride=1,
            padding=0):
        """
        Initializations
        :param input_size: (int, int): height, width tuple of the input
        :param input_channel: int: number of channels of the input
        :param hidden_channel: int: number of channels of the hidden state
        :param kernel_size: int: size of the filter
        :param stride: int: stride
        :param padding: int: width of the 0 padding
        """

        super(ConvLSTM, self).__init__()
        self.n_h, self.n_w = input_size
        self.n_c = input_channel
        self.hidden_channel = hidden_channel

        self.conv_xi = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xf = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xo = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_xg = nn.Conv2d(
            self.n_c,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hi = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hf = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_ho = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hg = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv_hi = nn.Conv2d(
            self.hidden_channel,
            self.hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, hidden_states):
        """
        Forward prop.
        :param x: input tensor of shape (n_batch, n_c, n_h, n_w)
        :param hidden_states: (tensor, tensor) for hidden and cell states.
                              Each of shape (n_batch, n_hc, n_hh, n_hw)
        :return: (hidden_state, cell_state)
        """

        hidden_state, cell_state = hidden_states

        xi = self.conv_xi(x)
        hi = self.conv_hi(hidden_state)
        xf = self.conv_xf(x)
        hf = self.conv_hf(hidden_state)
        xo = self.conv_xo(x)
        ho = self.conv_ho(hidden_state)
        xg = self.conv_xg(x)
        hg = self.conv_hg(hidden_state)

        i = torch.sigmoid(xi + hi)
        f = torch.sigmoid(xf + hf)
        o = torch.sigmoid(xo + ho)
        g = torch.tanh(xg + hg)

        cell_state = f * cell_state + i * g
        hidden_state = o * torch.tanh(cell_state)

        return hidden_state, cell_state

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda(),
                torch.zeros(batch_size, self.hidden_channel, self.n_h, self.n_w).cuda())


class ConvLSTMTwin(nn.Module):
    def __init__(self, input_size, hidden_dim, kernel_size, padding):
        """
        Init function.
        :param input_size: (int, int): input h, w
        """

        super(ConvLSTMTwin, self).__init__()

        self.pred_box_size = 1  # prediction looks at 2 * 1 + 1 box
        self.pred_box_w = 2 * self.pred_box_size + 1

        self.convlstm1 = ConvLSTM(
            input_size,
            3,
            hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.convlstm2 = ConvLSTM(
            input_size,
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.convlstm3 = ConvLSTM(
            input_size,
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        (self.hidden1, self.cell1), (self.hidden2, self.cell2), (self.hidden3, self.cell3) = \
            self.init_hidden(5)

        self.convOut = nn.Sequential(*[nn.Conv2d(16,64,kernel_size=8,stride=4,padding=2),nn.ReLU(),nn.Conv2d(64,256,kernel_size=8,stride=4,padding=2),
                                       nn.ReLU(),nn.Conv2d(256,1024,kernel_size=8,stride=4,padding=2)])





    def forward(self, X, hidden_states=None):
        m, Tx, n_c, n_h, n_w = X.shape

        meo_output = []
        aqi_output = []

        if hidden_states:
            (self.hidden1, self.cell1), (self.hidden2, self.cell2), (self.hidden3, self.cell3) = \
                hidden_states
        else:
            (self.hidden1, self.cell1), (self.hidden2, self.cell2), (self.hidden3, self.cell3) = \
                self.init_hidden(m)

        for t in range(Tx):
            xt = X[:, t, :, :, :]
            self.hidden1, self.cell1 = self.convlstm1(xt, (self.hidden1, self.cell1))
            self.hidden2, self.cell2 = self.convlstm2(self.hidden1, (self.hidden2, self.cell2))
            self.hidden3, self.cell3 = self.convlstm3(self.hidden2, (self.hidden3, self.cell3))



        hidden_states = (self.hidden1, self.cell1), (self.hidden2, self.cell2), (self.hidden3, self.cell3)

        return self.convOut(self.hidden3).view(-1,4096)

    def init_hidden(self, batch_size):
        return self.convlstm1.init_hidden(batch_size), \
               self.convlstm2.init_hidden(batch_size), \
               self.convlstm3.init_hidden(batch_size)



##############################################################################
# Twin Network
##############################################################################


class TwinNetwork(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, layer_dim=2, output_dim=4096):
        super(TwinNetwork, self).__init__()
        self.convlstmab= ConvLSTMTwin((128, 128), 16, 3, 1)

    def forward(self, X,Y):

        outA = self.convlstmab(X)
        outB = self.convlstmab(Y)


        #out = torch.exp(-(torch.abs(outA - outB)))
        return outA,outB





if __name__ == '__main__':

    from torch.autograd import Variable


    #writer = SummaryWriter("/home/ahmed/tensorboard/")

    device = torch.device('cuda:0')
    #netG =  Generator(256, 512, 8, channel_multiplier=2)
    #netD = Discriminator(256, channel_multiplier=2)
    #netG = ConVolutionNetwork()
    netG = TwinNetwork()
    net = netG.to(device)

    #sample_z = torch.randn(1, 512, device=device)
    #noise = mixing_noise(2, 512,0.9, device)
    inputA = torch.randn(1,100,3,128,128,device=device)
    inputB = torch.randn(1,89, 3, 128,128,device=device)

    outA,_ = net(inputA,inputB)
    print(outA.shape)
    #print(outB.shape)

    #writer.add_graph(net,  [noise])
    #writer.close()