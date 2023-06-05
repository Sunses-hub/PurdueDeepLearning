
import torch 
import torch.nn as nn
import pickle 
from torch.optim import Adam, RMSprop
import time 
from torch.utils.data import Dataset, DataLoader 
from torchvision.utils import make_grid, save_image 
import torchvision.transforms as tvt
from dataset import PizzaDataset
import torch.nn.functional as F
import os
import argparse
import random
import numpy as np 
from utils import calc_frechet

class Generator(nn.Module):
    def __init__(self, chn=3, bias=False, in_dim=100):
        super(Generator, self).__init__()
        # filter sizes: [1024, 512, 256, 128]
        self.chn = chn
        self.up_convs = nn.Sequential(
            # first up layer 
            nn.ConvTranspose2d(in_dim, 1024, 4, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(1024), 
            nn.ReLU(True), 
            # second up layer 
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(512), 
            nn.ReLU(True), 
            # third layer
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # fourth layer 
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.last_layer = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=bias),
                                        nn.Tanh())
    def forward(self, x):
        out = self.last_layer(self.up_convs(x))
        return out 

class Discriminator(nn.Module):
    def __init__(self, in_chn, neg_slope=0.2):
        super(Discriminator, self).__init__()
        # filter sizes: [128, 256, 512, 1024]

        self.in_chn = in_chn
        self.convs = nn.Sequential(
            # first down layer 
            nn.Conv2d(in_chn, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(neg_slope, inplace=True),
            # second down layer 
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(neg_slope, inplace=True),
            # third down layer 
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(neg_slope, inplace=True),
            # fourth down layer 
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(neg_slope, inplace=True),
            # fifth layer 
            nn.Conv2d(1024, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.convs(x)
        return out 

class Critic(nn.Module):
    def __init__(self, neg_slope=0.2, bias=False):
        super(Critic, self).__init__()
        # filter sizes: [128, 256, 512, 1024]
        self.convs = nn.Sequential(
            # first down layer 
            nn.Conv2d(3, 128, 4, stride=2, padding=1, bias=bias),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(neg_slope, inplace=True),
            # second down layer 
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=bias),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(neg_slope, inplace=True),
            # third down layer 
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=bias),
            #nn.BatchNorm2d(512), 
            nn.LeakyReLU(neg_slope, inplace=True),
            # fourth down layer 
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=bias),
            #nn.BatchNorm2d(1024),
            nn.LeakyReLU(neg_slope, inplace=True),
            # fifth layer 
            nn.Conv2d(1024, 1, 4, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        out = self.convs(x)
        return out 

class WGAN(object):
    def __init__(self, args):

        # set device  
        self.device = f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu" 

        # hyperparamters 
        self.beta1 = args.beta1
        self.beta2 = args.beta2 
        self.batch_size = args.batch_size 
        self.lr = args.lr
        self.gp = args.gp
        self.penalty = args.penalty if self.gp else None 
        self.clip = args.clip if not self.gp else None 
        self.zdim = args.noise_dim
        
        # generator & critic 
        self.generator = Generator(bias=args.bias, in_dim=args.noise_dim)
        self.critic = Critic(bias=args.bias)

        # train params 
        self.opt_c = Adam(params=self.critic.parameters(), 
                          lr=self.lr, betas=(self.beta1, self.beta2))
        self.opt_g = Adam(params=self.generator.parameters(),
                          lr=self.lr, betas=(self.beta1, self.beta2))
        self.c_iter = args.c_iter
        self.cl_iter = args.cl_iter 

    def train(self, train_loader, pre_trained=False, num_epochs=30):
        # that's where images will be saved 
        if not os.path.exists("wgan_fake_samples"):
            os.mkdir("wgan_fake_samples")
        print(f"training is running on {self.device}!")

        # histories 
        metrics = {
            "wasserstein" : [],
            "critic loss" : [],
            "gen loss" : []
        }

        # put the models to device 
        self.critic = self.critic.to(self.device)
        self.generator = self.generator.to(self.device)

        if pre_trained == False:
            self.critic.apply(weight_inits)
            self.generator.apply(weight_inits)
        else:
            self.critic.load_state_dict(torch.load("best_wgan_critic"))
            self.generator.load_state_dict(torch.load("best_wgan_generator"))

        # fixed noise 
        fixed_noise = torch.FloatTensor(self.batch_size, self.zdim, 1, 1).normal_(0, 1).to(self.device)
        # targets 
        one = torch.FloatTensor([1]).to(device=self.device)
        mone = torch.FloatTensor([-1]).to(device=self.device)
        # loss variables 
        gen_loss = 0 
        critic_loss = 0 
        wass_dist = 0
        # number of generator iterations 
        gen_iter = 0 
        for epoch in range(num_epochs):
            data_iter = iter(train_loader)
            idx = 0 # number of iteration in current epoch 
            while idx < len(train_loader):
                # activate weights of the critic
                for param in self.critic.parameters():
                    param.requires_grad = True 

                # that was 500
                c_iter = self.cl_iter if gen_iter < 25 or gen_iter % 500 == 0 else self.c_iter
                c_idx = 0 # number of critic iteration
                # train critic 
                while c_idx < c_iter and idx < len(train_loader):
                    # update counters 
                    c_idx += 1 
                    idx += 1 
                    # if not doing gradient penalty, then do gradient clipping 
                    if not self.gp:
                        for param in self.critic.parameters():
                            param.data.clamp_(-self.clip, self.clip)

                    # calculate real loss 
                    imgs, _ = next(data_iter)
                    imgs = imgs.to(self.device)

                    self.critic.zero_grad()
                    # calculate gradients for real part 
                    real_loss = self.critic(imgs).mean(dim=0).view(1)
                    real_loss.backward(mone)

                    # calculate gradients for fake part 
                    z = torch.randn(imgs.shape[0], self.zdim, 1, 1, device=self.device)
                    fake_imgs = self.generator(z) 
                    fake_loss = self.critic(fake_imgs).mean(dim=0).view(1)
                    fake_loss.backward(one)

                    # gradient penalty term 
                    if self.gp:
                        ratio = torch.FloatTensor(imgs.shape[0], 1, 1, 1).uniform_(0, 1).to(self.device)
                        # interpolated distribution 
                        int_dist = ratio * imgs + (1 - ratio) * fake_imgs.detach()
                        int_dist.requires_grad = True
                        c_out = self.critic(int_dist)
                        # calculate the gradient for soft constraint
                        c_grads = torch.autograd.grad(c_out, int_dist, torch.ones(c_out.size(), device=self.device),
                                                    create_graph=True, retain_graph=True)[0]
                        #print("LİMBASS CENNETİ:", c_grads.shape)
                        c_grads = c_grads.view(c_grads.shape[0], -1)
                        penalty = self.penalty * ((c_grads.norm(2, dim=1) - 1) ** 2).mean()
                        penalty.backward()

                    wass_dist = real_loss - fake_loss
                    loss_critic = -wass_dist
                    self.opt_c.step()

                # deactivate the critic weights 
                for param in self.critic.parameters():
                    param.requires_grad = False

                # train generator 
                self.generator.zero_grad()
                z = torch.randn(imgs.shape[0], self.zdim, 1, 1, device=self.device)
                fake_imgs = self.generator(z)
                loss_gen = self.critic(fake_imgs).mean().view(1)
                loss_gen.backward(mone) 
                gen_loss = -loss_gen
                self.opt_g.step()
                gen_iter += 1

                # save losses 
                print(f"[{epoch+1}/{num_epochs}][{idx}/{len(train_loader)}][{gen_iter}] loss_C: {wass_dist.item():.4f}, loss_G: {gen_loss.item():.4f} wass dist: {wass_dist.item():.4f}")
                metrics["wasserstein"].append(wass_dist.item())
                metrics["critic loss"].append(loss_critic.item())
                metrics["gen loss"].append(gen_loss.item())

                # save the results per 500 generator iteration
                if gen_iter % 500 == 0:
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                        fake = fake.mul(0.5).add(0.5)
                        save_image(fake, fp=f"wgan_fake_samples/{gen_iter//500}.jpg", nrow=int(self.batch_size**0.5))

        # save the results
        pickle.dump(metrics, open("wgan_results.pkl", "wb"))
        torch.save(self.generator.state_dict(), "best_wgan_generator")
        torch.save(self.critic.state_dict(), "best_wgan_critic")
        print("results and generator model are saved!")
        return metrics


class GAN(object):
    def __init__(self, args):

        # set device  
        self.device = f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu" 

        # hyperparamters 
        self.beta1 = args.beta1
        self.beta2 = args.beta2 
        self.batch_size = args.batch_size 
        self.lr = args.lr
        self.zdim = args.noise_dim
        
        # generator & discriminator  
        self.generator = Generator(in_dim=args.noise_dim)
        self.discriminator = Discriminator(in_chn=3)

        # train params 
        self.opt_d = Adam(params=self.discriminator.parameters(), 
                          lr=self.lr, betas=(self.beta1, self.beta2))
        self.opt_g = Adam(params=self.generator.parameters(),
                          lr=self.lr, betas=(self.beta1, self.beta2))

    def train(self, train_loader, num_epochs):

        # that's where images will be saved 
        if not os.path.exists("gan_fake_samples"):
            os.mkdir("gan_fake_samples")
        print(f"training is running on {self.device}!")

        # put the models to device 
        self.discriminator.to(self.device)
        self.generator.to(self.device)
        
        # intialize weights 
        self.discriminator.apply(weight_inits)
        self.generator.apply(weight_inits)

        # BCE loss 
        criterion = nn.BCELoss()

        target_real = torch.ones(self.batch_size, device=self.device)
        target_fake = torch.zeros(self.batch_size, device=self.device)

        losses = {"bce_loss" : [],
                  "g_loss" : [],
                  "d_loss" : []}
                
        mean_probs = {"real_probs" : [],
                      "fake_probs" : [], 
                      "fake_probs_g" : []}

        # fixed noise 
        fixed_noise = torch.FloatTensor(self.batch_size, self.zdim, 1, 1).normal_(0, 1).to(self.device)

        counter = 0

        for epoch in range(num_epochs):

            # for each batch 
            for i, (imgs, _) in enumerate(train_loader, 1):

                if imgs.shape[0] != self.batch_size:
                    continue 

                counter += 1
                
                self.discriminator.zero_grad()
                # get batch 
                imgs = imgs.to(self.device)

                # calculate discriminator loss
                pred_probs_r = self.discriminator(imgs).view(-1)
                real_loss = criterion(pred_probs_r, target_real)
                real_loss.backward()

                z = torch.randn(self.batch_size, self.zdim, 1, 1, device=self.device)
                fake_imgs = self.generator(z)
                pred_probs_f = self.discriminator(fake_imgs.detach()).view(-1)
                fake_loss = criterion(pred_probs_f, target_fake)
                fake_loss.backward()
                # total discriminator loss (BCE loss)
                bce_loss = real_loss.item() + fake_loss.item()
                # update weights 
                self.opt_d.step()
                
                # train generator 
                self.generator.zero_grad()
                pred_probs_f2 = self.discriminator(fake_imgs).view(-1)
                gen_loss = criterion(pred_probs_f2, target_real)
                gen_loss.backward()
                self.opt_g.step()

                # report and save the losses 
                print(f"[{i + epoch * len(train_loader)}/{len(train_loader) * num_epochs} ITER] discriminator loss: {bce_loss:.4f}, generator loss: {gen_loss.item():.4f}")

                # save the losses to history 
                losses["d_loss"].append(bce_loss)
                losses["g_loss"].append(gen_loss.item())
                # save mean probabilities 
                mean_probs["real_probs"].append(pred_probs_r.mean().item())
                mean_probs["fake_probs"].append(pred_probs_f.mean().item())
                mean_probs["fake_probs_g"].append(pred_probs_f2.mean().item())

                if counter % 500 == 0:
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                        fake = fake.mul(0.5).add(0.5)
                        save_image(fake, fp=f"gan_fake_samples/{counter//500}.jpg", nrow=int(self.batch_size**0.5))
    
        # save the results to local disk 
        results = {"losses" : losses, 
                   "probs" : mean_probs}

        pickle.dump(results, open("gan_results.pkl", "wb"))
        torch.save(self.generator.state_dict(), "best_gan_generator")
        torch.save(self.discriminator.state_dict(), "best_gan_discriminator")
        print("results and generator model are saved!")
        return results

# generic functions

def weight_inits(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def save_imgs(generator, z, file_name="images.jpg", best_gen="best_generator"):
    # load best generator 
    generator.load_state_dict(torch.load(best_gen)) 
    generator = generator.to('cpu')
    generator.eval()
    imgs = generator(z)
    imgs = imgs.mul(0.5).add(0.5)
    #grid = make_grid(imgs, nrow=8)
    save_image(imgs, file_name, nrow=8)
    print("generated images are saved")

# test code 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # common parameters in train
    parser.add_argument("--batch_size", default=64, type=int, help="batch size for SGD")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--cuda_idx", default=0, type=int, help="cuda id")
    parser.add_argument("--num_epochs", default=30, type=int, help="number of epochs for GAN training")
    parser.add_argument("--pre_trained", default=False, type=eval, help="start from pre-trained weights")
    parser.add_argument("--bias", default=False, type=eval, help="activate bias in CNN layers")
    # optimizer 
    parser.add_argument("--beta1", default=0.5, type=float, help="beta 1 of ADAM")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta 2 of ADAM")
    parser.add_argument("--c_iter", default=5, type=int, help="num. of iteration for critic")
    parser.add_argument("--cl_iter", default=5, type=int, help="large number for iteration at the beginning")
    # (W)GAN train parameters
    parser.add_argument("--gp", default=False, type=eval, help="activate gradient penalty")
    parser.add_argument("--noise_dim", default=100, type=int, help="number of dimension of noise vector")
    parser.add_argument("--train_wgan", default=True, type=eval, help="train wgan or gan")
    # WGAN_C parameter 
    parser.add_argument("--clip", default=0.005, type=float, help="clipping bound")
    # WGAN_GP parameter 
    parser.add_argument("--penalty", default=10, type=float, help="gradient penalty for WGAN")
    
    args = parser.parse_args()

    """
    Recommended GAN parameters
    beta1           : 0.5
    beta2           : 0.999
    learning rate   : 1e-4
    batch size      : 64
    noise dim       : 100
    num_epochs      : 125
    """

    """
    Recommended WGAN GP parameters
    beta1           : 0.5
    beta2           : 0.999
    c_iter          : 5 
    cl_iter         : 5
    penalty         : 10 
    learning rate   : 1e-4
    batch size      : 64
    noise dim       : 100
    num_epochs      : 750
    """

    # for reproducible results 
    seed = 101
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    torch.autograd.set_detect_anomaly(True)

    if args.gp:
        print("gradient penalty is activated!")
    else:
        print("gradient clipping is activated!")
    # get the data
    train_data = PizzaDataset(path="pizzas", train=True)
    test_data = PizzaDataset(path="pizzas", train=False)
    print("data is loaded!")
    # get dataloaders 
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


    if not args.train_wgan:
        # train and test GAN
        model = GAN(args)
        print("start training...")
        results = model.train(train_loader, args.num_epochs)
        print("training is successful!")
        # save the losses 
        pickle.dump(results, open("gan_results.pkl", "wb"))
        print("results were saved!")
        z = torch.randn(16, 100, 1, 1)
        save_imgs(model.generator, z, file_name="GAN_imgs.jpg", best_gen="best_gan_generator")
    else:
        # train and test WGAN
        model = WGAN(args)
        print("start training...")
        #results = model.train(train_loader, num_epochs=args.num_epochs)
        print("training is successful!")
        # save the results 
        pickle.dump(results, open("wgan_results.pkl", "wb"))
        z = torch.randn(64, 100, 1, 1)
        save_imgs(model.generator, z, file_name="WGAN_imgs.jpg", best_gen="best_wgan_generator")


    # calculate FID 
    model.generator.load_state_dict(torch.load("best_wgan_generator"))
    fid = calc_frechet(model, train_data, size=1000)
    print("frechet value:", fid)





