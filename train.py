import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import ImageFolder
from CycleGAN import CycleGAN
from model import Generator, Discriminator
from dataload import DatasetPipeline
from utils import *
import os


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current Device: ", device)

    # Set args
    batch_size = 1
    max_epoch = 200
    lr = 2e-4
    trial = 2
    current_epoch = 10
    checkpoint_dir = f'./checkpoint/cp_{trial}_{current_epoch}.pt'
    result_dir = f'./result/cp_{trial}'
    rootA = '../../Datasets/horse2zebra/trainA'
    rootB = '../../Datasets/horse2zebra/trainB'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # Define Generator and Discriminator
    genAB = Generator(3).to(device)
    genBA = Generator(3).to(device)

    discA = Discriminator(3).to(device)
    discB = Discriminator(3).to(device)

    if os.path.exists(checkpoint_dir):
        state_dict = torch.load(checkpoint_dir)
        genAB.load_state_dict(state_dict['genAB'])
        genBA.load_state_dict(state_dict['genBA'])

        discA.load_state_dict(state_dict['discA'])
        discB.load_state_dict(state_dict['discB'])

        print("Checkpoint Loaded")
    else:
        genAB.apply(weights_init)
        genBA.apply(weights_init)

        discA.apply(weights_init)
        discB.apply(weights_init)

        print("New Model")

    
    # Define Transform
    tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize([256, 256])
            ])

    # Define Dataloder
    dataset = DatasetPipeline(rootA=rootA, rootB=rootB)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)


    gan_trainer = CycleGAN(genAB=genAB,
                      genBA=genBA,
                      discA=discA,
                      discB=discB,
                      max_epoch=max_epoch,
                      batch_size=batch_size,
                      lr=lr,
                      dataloader=dataloader,
                      trial=trial,
                      current_epoch=current_epoch,
                      tf=tf,
                      device=device
                      )
    
    gan_trainer.train()



if __name__=='__main__':
    main()


