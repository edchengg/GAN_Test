import argparse
import math
import os
import torch
from models.GAN import GAN
from models.LSGAN import LSGAN
from models.WGAN import WGAN
from models.WGAN_GP import WGAN_GP


parser = argparse.ArgumentParser(
    description='Generative Adverserial Models'
)

parser.add_argument('--epoch', type=int, default=50,
                    help='number of epoch')
parser.add_argument('--model_name', type=str, default='GAN',
                    choices=['GAN','LSGAN','WGAN','WGAN-GP'],
                    help='The type of GAN')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--batch_size', type=int, default=64,
                    help='The size of batch')
parser.add_argument('--input_size', type=int, default=28,
                    help='The size of input image')
parser.add_argument('--save_dir', type=str, default='save',
                    help='Directroy name to save the mdoel')
parser.add_argument('--LR_G', type=float, default=0.0002,
                    help='Learning rate for generator optimizer')
parser.add_argument('--LR_D', type=float, default=0.0002,
                    help='Learning rate for discriminator optimizer')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='Adam beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='Adam beta2')
parser.add_argument('--CUDA', type=bool, default=False,
                    help='Use GPU')

args = parser.parse_args()


if args.model_name == 'GAN':
    model = GAN(args)

if args.model_name == 'LSGAN':
    model = LSGAN(args)
if args.model_name == 'WGAN':
    model = WGAN(args)
if args.model_name == 'WGAN-GP':
    model = WGAN_GP(args)

print('Training Starts')
model.train()
print('Training Finish')
