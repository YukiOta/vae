from __future__ import print_function

import argparse
import os
import time
import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from util import *

parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--resume', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--img-size', type=int, default=128, metavar='N',
                    help='input image size for the model (default: 128)')
parser.add_argument('--filter-size', type=int, default=128, metavar='N',
                    help='filter size for the model (default: 128)')
parser.add_argument('--latent-size', type=int, default=500, metavar='N',
                    help='latent size for the model (default: 500)')
parser.add_argument('--model-dirname', type=str, default='models', metavar='N',
                    help='dirname of model dirctory (default: "models")')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# batch size
batch_size = args.batch_size
im_size = args.img_size
filter_size = args.filter_size
latent_size = args.latent_size

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = range(2080)
test_loader = range(40)

totensor = transforms.ToTensor()
def load_batch(istrain):
    # data_loaderを用いる新しいversion
    if istrain:
        template = '../data/PV_IMAGE/201705'
    else:
        template = '../data/vae_solar/test/'
        # template = '../data/PV_IMAGE/201706'
    # l = [str(batch_idx*batch_size + i).zfill(6) for i in range(batch_size)]
    # load dataset
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(template, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # images, labels = iter(data_loader).next()
    return data_loader

class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4*4)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        eps = eps.to(device)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar


model = VAE(nc=3, ngf=filter_size, ndf=filter_size, latent_variable_size=latent_size).to(device)

if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(epoch):
    model.train()
    train_loss = 0
    data_loader = load_batch(istrain=True)
    # for batch_idx in train_loader:
    for i, datas in enumerate(data_loader):
        # data = load_batch(batch_idx, True)
        data, _ = datas
        data = data.to(device)
        # data = Variable(data)
        # if args.cuda:
        #     data = data.cuda()
        # 勾配の初期化
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        # train_loss += loss.data[0]
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, len(data_loader),
                100. * i / len(data_loader),
                loss.item() ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(data_loader))))
    return train_loss / (len(data_loader))

def test(epoch):
    model.eval()
    test_loss = 0
    # data loader
    data_loader = load_batch(istrain=False)

    # for batch_idx in test_loader:
    for i, datas in enumerate(data_loader):
        # data = load_batch(batch_idx, False)
        data, _ = datas
        data = data.to(device)
        # data = Variable(data, volatile=True)
        # if args.cuda:
        #     data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).item() 

        # torchvision.utils.save_image(data.data, './imgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
        torchvision.utils.save_image(data.data, os.path.join('./imgs', dir_name, 'Epoch_{}_data.jpg'.format(epoch)), nrow=8, padding=2)
        torchvision.utils.save_image(recon_batch.data, os.path.join('./imgs', dir_name, 'Epoch_{}_recon.jpg'.format(epoch)), nrow=8, padding=2)

    test_loss /= (len(data_loader))
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def perform_latent_space_arithmatics(items): # input is list of tuples of 3 [(a1,b1,c1), (a2,b2,c2)]
    load_last_model()
    model.eval()
    data = [im for item in items for im in item]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    # data = Variable(data, volatile=True)
    data = data.to(device)
    # if args.cuda:
    #     data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it, it)
    zs = []
    numsample = 11
    for i,j,k in z:
        for factor in np.linspace(0,1,numsample):
            zs.append((i-j)*factor+k)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data.split(1))
    it2 = [iter(recon.split(1))]*numsample
    result = zip(it1, it1, it1, *it2)
    result = [im for item in result for im in item]

    result = torch.cat(result, 0)
    torchvision.utils.save_image(result.data, './imgs/vec_math.jpg', nrow=3+numsample, padding=2)


def latent_space_transition(items): # input is list of tuples of  (a,b)
    load_last_model()
    model.eval()
    data = [im for item in items for im in item[:-1]]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    # data = Variable(data, volatile=True)
    data = data.to(device)
    # if args.cuda:
    #     data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it)
    zs = []
    numsample = 11
    for i,j in z:
        for factor in np.linspace(0,1,numsample):
            zs.append(i+(j-i)*factor)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data.split(1))
    it2 = [iter(recon.split(1))]*numsample
    result = zip(it1, it1, *it2)
    result = [im for item in result for im in item]

    result = torch.cat(result, 0)
    torchvision.utils.save_image(result.data, './imgs/trans.jpg', nrow=2+numsample, padding=2)


def rand_faces(num=5):
    load_last_model()
    model.eval()
    z = torch.randn(num*num, model.latent_variable_size)
    z = z.to(device)
    # z = Variable(z, volatile=True)
    # if args.cuda:
    #     z = z.cuda()
    recon = model.decode(z)
    torchvision.utils.save_image(recon.data, './imgs/rand_faces.jpg', nrow=num, padding=2)

def load_last_model(dir_name):
    # dir_path内に含まれる.pthファイルから最後のエポックのものを持ってくる
    models = glob(os.path.join('./models/', dir_name, '*.pth'))  
    print(models)
    model_ids = [(int(f.split('_')[1]), f) for f in models]
    start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
    print('Last checkpoint: ', last_cp)
    model.load_state_dict(torch.load(last_cp))
    return start_epoch, last_cp

def resume_training(dir_name):
    start_epoch, _ = load_last_model(dir_name)
    print('resume training from Epoch %d' % (start_epoch+1))

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        target_dir = os.path.join(
            './models/',
            dir_name,
            'Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss)
            )
        torch.save(model.state_dict(), target_dir)

def last_model_to_cpu():
    _, last_cp = load_last_model()
    model.cpu()
    torch.save(model.state_dict(), './models/cpu_'+last_cp.split('/')[-1])

def data_rename(dataPath):
    # trainPath = os.path.join(dataPath, "train")
    # testPath = os.path.join(dataPath, "test")
    ls_file = glob(os.path.join(dataPath, "*.png"))
    ls_file.sort()
    print(ls_file[0].split("/")[-1])
    if ls_file[0].split("/")[-1] == "img000000.png":
        print("ready")
    else:
        for count, filename in enumerate(ls_file):
            os.rename(ls_file[count], os.path.join(dataPath, "img{0:06d}".format(count) + os.path.splitext(ls_file[count])[1]))

def train_from_scratch():
    print('start training from scrach')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        target_dir = os.path.join(save_dir, 'Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss))
        torch.save(model.state_dict(), target_dir)

if __name__ == '__main__':
    # data_rename("../data/vae_solar/train/train")
    # data_rename("../data/vae_solar/test/test")
    if args.resume:
        # 学習済みのモデルを使って学習を再開する
        dir_name = args.model_dirname
        resume_training(dir_name)
    else:
        # 最初から学習する
        time_now = datetime.datetime.now().strftime('%Y%m%d%H%M')
        save_dir = os.path.join('./models/', time_now)
        dir_name = time_now
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_from_scratch()

    # last_model_to_cpu()
    # load_last_model()
    # rand_faces(10)
    # da = load_pickle(test_loader[0])
    # da = da[:120]
    # it = iter(da)
    # l = zip(it, it, it)
    # # latent_space_transition(l)
    # perform_latent_space_arithmatics(l)
