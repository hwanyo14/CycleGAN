import torch
from torch import nn
from torch.autograd import Variable
from PIL import Image
import random



# Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_result_images(genAB, genBA, tf):
    imgA_path = '../../Datasets/horse2zebra/test_pair/A.jpg'
    imgB_path = '../../Datasets/horse2zebra/test_pair/B.jpg'

    imgA = Image.open(imgA_path).convert("RGB")
    imgB = Image.open(imgB_path).convert("RGB")

    imgA = tf(imgA).cuda()
    imgB = tf(imgB).cuda()

    imgA = torch.unsqueeze(imgA, 0)
    imgB = torch.unsqueeze(imgB, 0)

    genAB.eval()
    genBA.eval()

    with torch.no_grad():
        fake_A = genBA(imgB)    # fake horse
        fake_B = genAB(imgA)    # fake zebra

    imgA = imgA * 0.5 + 0.5
    imgB = imgB * 0.5 + 0.5
    fake_A = fake_A * 0.5 + 0.5
    fake_B = fake_B * 0.5 + 0.5

    real_data = torch.cat([imgA, imgB], 0)
    fake_data = torch.cat([fake_B, fake_A], 0)

    return torch.cat([real_data, fake_data], 0)


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))