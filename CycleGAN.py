import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.autograd import Variable
from utils import get_result_images
import itertools



class CycleGAN():
    def __init__(self, 
                 genAB,
                 genBA,
                 discA,
                 discB, 
                 max_epoch,
                 batch_size,
                 lr,
                 dataloader,
                 trial,
                 current_epoch,
                 tf,
                 device
                 ):
        
        self.genAB = genAB
        self.genBA = genBA

        self.discA = discA
        self.discB = discB

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.dataloader = dataloader
        self.trial = trial
        self.current_epoch= current_epoch
        self.tf = tf
        self.device = device
        self.checkpoint_save_dir = './checkpoint'

        self.lambda_id = 2
        self.lambda_cyc = 10

        self.g_optim = torch.optim.Adam(itertools.chain(self.genAB.parameters(), self.genBA.parameters()),
                                        lr=lr, betas=(0.5, 0.999))
        self.dA_optim = torch.optim.Adam(self.discA.parameters(), lr=lr, betas=(0.5, 0.999))
        self.dB_optim = torch.optim.Adam(self.discB.parameters(), lr=lr, betas=(0.5, 0.999))


        self.mse_loss = nn.MSELoss()
        self.id_loss = nn.L1Loss()
        self.cyc_loss = nn.L1Loss()

        self.real_label = torch.ones((self.batch_size, 1, 16, 16), device=self.device)
        self.fake_label = torch.zeros((self.batch_size, 1, 16, 16), device=self.device)


    def train(self):
        for epoch in range(1, self.max_epoch+1):
            if self.current_epoch >= epoch:
                continue
            self.genAB.train()
            self.genBA.train()

            self.discA.train()
            self.discB.train()

            for idx, (imgA, imgB) in enumerate(tqdm(self.dataloader)):
                # Real Data
                real_A = Variable(imgA, requires_grad=True).to(self.device)
                real_B = Variable(imgB, requires_grad=True).to(self.device)

                # ===== Train Generator =====
                self.g_optim.zero_grad()

                # Identity Loss
                fake_B = self.genBA(real_A)
                fake_A = self.genAB(real_B)

                id_loss_B = self.id_loss(fake_B, real_A)
                id_loss_A = self.id_loss(fake_A, real_B)

                id_loss = (id_loss_A + id_loss_B) / 2.
                
                # GAN Loss
                fake_B = self.genAB(real_A)
                fake_A = self.genBA(real_B)

                fake_B_pred = self.discB(fake_B)
                fake_A_pred = self.discA(fake_A)

                gan_loss_B = self.mse_loss(fake_B_pred, self.real_label)
                gan_loss_A = self.mse_loss(fake_A_pred, self.real_label)

                gan_loss = (gan_loss_A + gan_loss_B) / 2.

                # Cycle Loss
                recov_A = self.genBA(fake_B)
                recov_B = self.genAB(fake_A)

                cyc_loss_A = self.cyc_loss(recov_A, real_A)
                cyc_loss_B = self.cyc_loss(recov_B, real_B)

                cyc_loss = (cyc_loss_A + cyc_loss_B) / 2.

                g_loss = gan_loss + self.lambda_id*id_loss + self.lambda_cyc*cyc_loss
                g_loss.backward()
                self.g_optim.step()

                # ========== Train Discriminator A ==========
                self.discA.zero_grad()

                real_loss_a = self.mse_loss(self.discA(real_A), self.real_label)
                fake_loss_a = self.mse_loss(self.discA(fake_A.detach()), self.fake_label)

                disc_loss_A = (real_loss_a + fake_loss_a) / 2

                disc_loss_A.backward()
                self.dA_optim.step()


                # ========== Train Discriminator B ==========
                self.discB.zero_grad()

                real_loss_b = self.mse_loss(self.discB(real_B), self.real_label)
                fake_loss_b = self.mse_loss(self.discB(fake_B.detach()), self.fake_label)

                disc_loss_B = (real_loss_b + fake_loss_b) / 2

                disc_loss_B.backward()
                self.dB_optim.step()

                print(f'Epoch {epoch}/{self.max_epoch}, G_Loss: {g_loss.data}, D_A_Loss: {disc_loss_A.data}, D_B_Loss: {disc_loss_B.data}')

            torch.save({
                'genAB': self.genAB.state_dict(),
                'genBA': self.genBA.state_dict(),
                'discA': self.discA.state_dict(),
                'discB': self.discB.state_dict(),
            }, f'{self.checkpoint_save_dir}/cp_{self.trial}_{epoch}.pt')

            result = get_result_images(self.genAB, self.genBA, self.tf)
            grid = make_grid(result, nrow=2)
            save_image(grid, f'./result/cp_{self.trial}/{str(epoch).zfill(3)}.jpg')


                
        






