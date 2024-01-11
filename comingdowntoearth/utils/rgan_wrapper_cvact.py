import os

from itertools import chain

import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Parameter

from .base_wrapper import BaseModel
from ..networks.c_gan import *

class RGANWrapper(BaseModel):

    def __init__(self, opt, log_file, net_G, net_D, net_R, infoNCE):
        BaseModel.__init__(self, opt, log_file)
        self.optimizers = []
        self.ret_best_acc = 0.0
        self.generator = net_G
        self.retrieval = net_R
        self.discriminator = net_D
        self.criterion = GANLoss(opt.gan_loss).to(opt.device)
        self.criterion_l1 = torch.nn.L1Loss()
        self.infoNCE = infoNCE
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Optimizers
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
        self.optimizer_R = torch.optim.Adam(self.retrieval.parameters(), lr=opt.lr_r, betas=(opt.b1, opt.b2))

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.optimizers.append(self.optimizer_R)
        self.scaler_G = GradScaler(init_scale=2.**10)
        self.scaler_D = GradScaler(init_scale=2.**10)
        self.scaler_R = GradScaler(init_scale=2.**10)
        self.clip_grad = 100.  # None | float

        self.load_networks()

    def forward(self):
        self.fake_street, self.residual = self.generator(self.satellite)

    def backward_D(self):

        self.optimizer_D.zero_grad()

        # Fake discriminator train
        fake_satellitestreet = torch.cat((self.satellite, self.fake_street), 1)
        fake_pred = self.discriminator(fake_satellitestreet.detach())
        self.d_loss_fake = self.criterion(fake_pred, False)

        # Real discriminator train
        real_satellitestreet = torch.cat((self.satellite, self.street), 1)
        real_pred = self.discriminator(real_satellitestreet)
        self.d_loss_real = self.criterion(real_pred, True)

        self.d_loss = (self.d_loss_real + self.d_loss_fake) * 0.5


    def backward_R(self):
        self.optimizer_R.zero_grad()

        self.fake_street_out, self.street_out = self.retrieval(self.street, self.residual.detach())

        self.r_loss = self.infoNCE(self.fake_street_out, self.street_out, self.logit_scale)

    def backward_G(self):
        self.optimizer_G.zero_grad()

        fake_satellitestreet = torch.cat((self.satellite, self.fake_street), 1)
        fake_pred = self.discriminator(fake_satellitestreet)
        self.gan_loss = self.criterion(fake_pred, True)

        self.fake_street_out, self.street_out = self.retrieval(self.street, self.residual)
        self.ret_loss = self.infoNCE(self.fake_street_out, self.street_out, self.logit_scale)

        self.g_l1 = self.criterion_l1(self.fake_street, self.street) * self.opt.lambda_l1
        self.g_loss = self.gan_loss + self.ret_loss + self.g_l1

    def optimize_parameters(self):
        # TODO: use the original model in this wrapper on its own and see if I can get the same accuracy
        # TODO: Try the original image size and see if I can get the same accuracy
        with autocast():
            self.forward()
            # update D
            self.set_requires_grad(self.discriminator, True)
            self.backward_D()
        self.scaler_D.scale(self.d_loss).backward()
        self.scaler_D.unscale_(self.optimizer_D)
        # torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), self.clip_grad)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad)
        self.scaler_D.step(self.optimizer_D)
        self.scaler_D.update()
        if self.scheduler_D is not None:
            self.scheduler_D.step()

        with autocast():
            # update R
            self.set_requires_grad(self.discriminator, False)
            self.set_requires_grad(self.retrieval, True)
            self.backward_R()
        self.scaler_R.scale(self.r_loss).backward()
        self.scaler_R.unscale_(self.optimizer_R)
        # torch.nn.utils.clip_grad_value_(self.retrieval.parameters(), self.clip_grad)
        torch.nn.utils.clip_grad_norm_(self.retrieval.parameters(), self.clip_grad)
        self.scaler_R.step(self.optimizer_R)
        self.scaler_R.update()
        if self.scheduler_R is not None:
            self.scheduler_R.step()

        with autocast():
            # update G
            self.set_requires_grad(self.retrieval, False)
            self.backward_G()
        self.scaler_G.scale(self.g_loss).backward()
        self.scaler_G.unscale_(self.optimizer_G)
        # torch.nn.utils.clip_grad_value_(self.generator.parameters(), self.clip_grad)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip_grad)
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()
        if self.scheduler_G is not None:
            self.scheduler_G.step()

    def predict(self, street, satellite):
        self.street = street
        self.satellite = satellite
        self.forward()
        self.fake_street_out, self.street_out = self.retrieval(self.street, self.residual)
        return self.street_out, self.fake_street_out

    def __call__(self, street, satellite):
        self.street = street
        self.satellite = satellite
        self.optimize_parameters()
        return self.street_out, self.fake_street_out


    def save_networks(self, epoch, out_dir, last_ckpt=False, best_acc=None, is_best=False):
        ckpt = {'last_epoch': epoch,
                'best_acc': best_acc,
                'generator_model_dict': self.generator.state_dict(),
                'optimizer_G_dict': self.optimizer_G.state_dict(),
                'discriminator_model_dict': self.discriminator.state_dict(),
                'optimizer_D_dict': self.optimizer_D.state_dict(),
                'retriebal_model_dict': self.retrieval.state_dict(),
                 'optimizer_R_dict': self.optimizer_R.state_dict(),
                }

        if last_ckpt:
            ckpt_name = 'rgan_last_ckpt.pth'
        elif is_best:
            ckpt_name = 'rgan_best_ckpt.pth'
        else:
            ckpt_name = 'rgan_ckpt_ep{}.pth'.format(epoch + 1)
        ckpt_path = os.path.join(out_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_networks(self):
        if self.opt.rgan_checkpoint is None:
            return

        ckpt_path = self.opt.rgan_checkpoint
        ckpt = torch.load(ckpt_path)
        self.ret_best_acc = ckpt['best_acc']

        # Load net state
        generator_dict = ckpt['generator_model_dict']
        discriminator_dict = ckpt['discriminator_model_dict']
        retrieval_dict = ckpt['retriebal_model_dict']

        self.generator.load_state_dict(generator_dict, strict=False)
        self.optimizer_G = ckpt['optimizer_G_dict']

        self.discriminator.load_state_dict(discriminator_dict)
        self.optimizer_D = ckpt['optimizer_D_dict']

        self.retrieval.load_state_dict(retrieval_dict)
        self.optimizer_R = ckpt['optimizer_R_dict']

    def parameters(self, recurse: bool = True):
        # Chain the parameters of generator, retrieval and discriminator
        return chain(self.generator.parameters(), self.retrieval.parameters(), self.discriminator.parameters())

    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
        # Chain the parameters of generator, retrieval and discriminator
        return chain(self.generator.named_parameters(prefix, recurse, remove_duplicate),
                     self.retrieval.named_parameters(prefix, recurse, remove_duplicate),
                     self.discriminator.named_parameters(prefix, recurse, remove_duplicate))
    def train(self):
        self.generator.train()
        self.retrieval.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.retrieval.eval()
        self.discriminator.eval()









