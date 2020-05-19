import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class SiameseModel(BaseModel):
    def name(self):
        return 'SiameseModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='Video')


        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['G']
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.visual_names = ['A1V','A2V','B1V','B2V']

        #self.avg_paired = 0
        #self.avg_unpaired = 0
        #self.number=1

        # specify the images you want to save/display. The program will call base_model.get_current_visuals



        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        self.model_names_ = ['G']
        self.model_names = ['G']

        self.netG = networks.define_G('g', opt, self.gpu_ids).to(self.device)
        self.criterion = networks.ContrastiveLoss().to(self.device)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=0.001)
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.A1 = input['A1'].to(self.device)
        self.A2 = input['A2'].to(self.device)
        self.B1 = input['B1'].to(self.device)
        self.B2 = input['B2'].to(self.device)

        #self.A11 = input['A1' ].to(self.device)
        #self.A22 = input['A2' ].to(self.device)
        #self.B11 = input['B1' ].to(self.device)
        #self.B22 = input['B2' ].to(self.device)

        self.image_paths = input['A_paths']
        self.A1V = torch.zeros(64,3,128,128).to(self.device)
        b,c,w,h = self.A1[0].shape
        if b>64:
            self.A1V = self.A1[0][0:64,:,:,:]
        else:
            self.A1V[0:b,:,:,:] = self.A1[0]

        self.A2V = torch.zeros(64,3,128,128).to(self.device)
        b, c, w, h = self.A2[0].shape
        if b > 64:
            self.A2V = self.A2[0][0:64, :, :, :]
        else:
            self.A2V[0:b, :, :, :] = self.A2[0]

        self.B1V = torch.zeros(64,3,128,128).to(self.device)
        b, c, w, h = self.B1[0].shape
        if b > 64:
            self.B1V = self.B1[0][0:64, :, :, :]
        else:
            self.B1V[0:b, :, :, :] = self.B1[0]

        self.B2V = torch.zeros(64,3,128,128).to(self.device)
        b, c, w, h = self.B2[0].shape
        if b > 64:
            self.B2V = self.B2[0][0:64, :, :, :]
        else:
            self.B2V[0:b, :, :, :] = self.B2[0]


    def forward(self):

        self.outA1,self.outA2 = self.netG(self.A1,self.A2)
        self.outB1,self.outB2 = self.netG(self.B1, self.B2)

        #if torch.dist(self.outA1,self.outA2,2).item()!=0:
            #self.avg_paired = torch.dist(self.outA1,self.outA2,2).item()
            #self.avg_unpaired = torch.dist(self.outB1,self.outB2,2).item()
            #self.number+=1

        #print('average paired', self.avg_paired)
        #print('average unpaired', self.avg_unpaired)


    def backward_G(self):

        self.loss_G = self.criterion(self.outA1,self.outA2,0) + self.criterion(self.outB1,self.outB2,1)

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        #networks.requires_grad(self.netG, True)
        self.netG.zero_grad()
        self.backward_G()
        nn.utils.clip_grad_norm_(self.netG.parameters(), 1e-5)
        self.optimizer_G.step()
