import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from models.modules import *
from dataloader import *
from tensorboardX import SummaryWriter

class LSGAN(object):
    def __init__(self, args):

        self.z_dim = 62
        self.sample_num = 100
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.CUDA = args.CUDA
        self.epoch = args.epoch
        self.model_name = args.model_name
        self.save_dir = args.save_dir
        # self.result_dir = args.result_dir

        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        self.G = Generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        # LSGAN: remove the sigmoid layer from the discriminator
        self.D = Discriminator(input_channel=data.shape[1], output_dim=1, input_size=self.input_size, sigmoid=False)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.LR_G, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.LR_D, betas=(args.beta1, args.beta2))

        self.writer = SummaryWriter(self.model_name)
        if self.CUDA:
            self.G.cuda()
            self.D.cuda()
            # LSGAN: L2 loss
            self.L2_loss = nn.MSELoss().cuda()
        else:
            self.L2_loss = nn.MSELoss()

        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.CUDA:
            self.sample_z_ = self.sample_z_.cuda()


    def train(self):
        self.real_label, self.fake_label = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.CUDA:
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()

        self.D.train()
        self.num_iteration = 0
        self.visualize_results(1)
        for epoch in range(self.epoch):
            self.G.train()
            for idx, (x, _) in enumerate(self.data_loader):
                if idx == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z = torch.rand((self.batch_size, self.z_dim))
                if self.CUDA:
                    z = z.cuda()
                    x = x.cuda()

                ''' ================== Train Discriminator ============='''

                self.D_optimizer.zero_grad()

                D_real = self.D(x)
                # LSGAN: L2 loss
                D_real_loss = self.L2_loss(D_real, self.real_label)

                x_fake = self.G(z)
                D_fake = self.D(x_fake)
                D_fake_loss = self.L2_loss(D_fake, self.fake_label)

                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                self.D_optimizer.step()

                ''' ================== Train Generator ============='''
                self.G_optimizer.zero_grad()

                x_fake = self.G(z)
                D_fake = self.D(x_fake)
                # LSGAN: L2 loss
                G_loss = self.L2_loss(D_fake, self.real_label)
                G_loss.backward()

                self.G_optimizer.step()

                # ======== 1 iteration ======== #
                self.num_iteration += 1

                if idx % 10 == 0:
                    print("Iteration: [%10d] D_loss : %.8f, G_loss: %.8f"%
                          ((self.num_iteration), (D_loss.item()), (G_loss.item())))


                    # Summary loss
                    self.writer.add_scalars('data/loss', {'Generator': G_loss.item(),
                                                          'Discriminator': D_loss.item()},
                                            self.num_iteration)

            # ======= visualize ======== #
            self.visualize_results(epoch)
        self.save()

    def visualize_results(self, epoch):
        self.G.eval()
        x = self.G(self.sample_z_)
        x = vutils.make_grid(x, normalize=True, scale_each=True)
        self.writer.add_image('Image', x, epoch)



    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.mkdirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, '_G.bin'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, '_D.bin'))