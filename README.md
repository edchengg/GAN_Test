# GAN Experiments

Experiments with different GAN models.

## List of GAN
- GAN
- LSGAN
- WGAN
- WGAN-GP
- CycleGAN
- infoGAN

## Results

*Name* | *Epoch 1* | *Epoch 2* | *Epoch 3* | *Epoch 37* | *Epoch 50*
:---: | :---: | :---: | :---: | :---: | :---: |
GAN | <img src = 'figures/GAN/gan_0.png'> | <img src = 'figures/GAN/gan_1.png'>| <img src = 'figures/GAN/gan_2.png'>| <img src = 'figures/GAN/gan_36.png'> | <img src = 'figures/GAN/gan_49.png'>
LSGAN | <img src = 'figures/LSGAN/lsgan_1.png'> | <img src = 'figures/LSGAN/lsgan_2.png'>| <img src = 'figures/LSGAN/lsgan_3.png'>| <img src = 'figures/LSGAN/lsgan_36.png'> | <img src = 'figures/LSGAN/lsgan_49.png'>
WGAN | <img src = 'figures/WGAN/wgan_0.png'> | <img src = 'figures/WGAN/wgan_1.png'>| <img src = 'figures/WGAN/wgan_3.png'>| <img src = 'figures/WGAN/wgan_37.png'> | <img src = 'figures/WGAN/wgan_50.png'>
WGAN-GP | <img src = 'figures/WGAN-GP/wgan-gp4.png'> | <img src = 'figures/WGAN-GP/wgan-gp-5.png'>| <img src = 'figures/WGAN-GP/wgan-gp-8.png'>| <img src = 'figures/WGAN-GP/wgan-gp-39.png'> | <img src = 'figures/WGAN-GP/wgan-gp-50.png'>

*Name* | *Loss* | *Legend*
:---: | :---: | :---: |
GAN | <img src = 'figures/GAN/gan_loss.png'> | Dis(![#f03c15](https://placehold.it/15/f03c15/000000?text=+)), Gen(![#1589F0](https://placehold.it/15/1589F0/000000?text=+))
LSGAN| <img src = 'figures/LSGAN/lsgan_loss.png'> | Dis(![#f03c15](https://placehold.it/15/f03c15/000000?text=+)), Gen(![#1589F0](https://placehold.it/15/1589F0/000000?text=+))
WGAN| <img src = 'figures/WGAN/wgan_loss.png'> | Dis(![#1589F0](https://placehold.it/15/1589F0/000000?text=+)), Gen(![#F08C27](https://placehold.it/15/F08C27/000000?text=+))
WGAN-GP| <img src = 'figures/WGAN-GP/wgan-gp-loss.png'> | Dis(![#15c9f0](https://placehold.it/15/15c9f0/000000?text=+)), Gen(![#f23c6a](https://placehold.it/15/f23c6a/000000?text=+))

### CycleGAN
<img src = 'figures/CycleGAN/loss.png'>
Horse2Zebra:
<img src = 'figures/CycleGAN/horse1.png', width="300"> 
<img src = 'figures/CycleGAN/horse2.png', width="300"> 
<img src = 'figures/CycleGAN/horse3.png', width="300"> 


## TODO
- [x] WGAN-GP
- [x] CycleGAN
- [ ] infoGAN


