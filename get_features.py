import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from util import *

parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--img-size', type=int, default=128, metavar='N',
                    help='input image size for the model (default: 128)')
parser.add_argument('--model-path', type=str,
                    help='path for input learned model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# batch size
batch_size = args.batch_size
im_size = args.img_size
path = args.model_path

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
totensor = transforms.ToTensor()


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

        self.e2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.e3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.e4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.e5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf * 8)

        self.fc1 = nn.Linear(ndf * 8 * 4 * 4, latent_variable_size)
        self.fc2 = nn.Linear(ndf * 8 * 4 * 4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf * 8 * 2 * 4 * 4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf * 8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf * 8, ngf * 4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf * 4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf * 4, ngf * 2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf * 2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf * 2, ngf, 3, 1)
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
        h5 = h5.view(-1, self.ndf * 8 * 4 * 4)

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
        h1 = h1.view(-1, self.ngf * 8 * 2, 4, 4)
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


def main():
    # 潜在変数の数を変える
    model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500).to(device)
    if args.cuda:
        model.cuda()
    model.load_state_dict(torch.load(path))

    data_path = '../data/vae_solar/test'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_path, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    datX_all = []
    datY_all = []
    for i, data in enumerate(data_loader):
        print("%d / %d" % (i, len(data_loader)))
        images, labels = iter(data_loader).next()
        data = images.to(device)
        labels = labels.to(device)
        print("The number of data is %d" % (len(labels)))
        # labels = labels.data.numpy()

        recon_batch, mu, logvar = model(data)

        mu = mu.cpu()
        logvar = logvar.cpu()

        mu, logvar = mu.data.numpy(), logvar.data.numpy()

        # numpy配列の保存
        np.save("./results/datX_%d.npy" % (i), mu)
        np.save("./results/datY_%d.npy" % (i), labels)
        # datX_all.append(mu)
        # datY_all.append(labels)

    # numpy配列の保存
    # np.save("./datX.npy", datX_all)
    # np.save("./datY.npy", datY_all)


if __name__ == '__main__':
    main()
