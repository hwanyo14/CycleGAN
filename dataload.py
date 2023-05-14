import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image


class DatasetPipeline(Dataset):
    def __init__(self, rootA, rootB) -> None:
        '''
            For CMPFacade dataset
            real image data extension: jpg
            segmentation label data extension: png
        '''
        super().__init__()
        self.all_imgA = sorted(glob(f'{rootA}/*.jpg'))
        self.all_imgB = sorted(glob(f'{rootB}/*.jpg'))

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize([256, 256])
        ])

    def __getitem__(self, index):
        imgA = Image.open(self.all_imgA[index]).convert('RGB')
        imgB = Image.open(self.all_imgB[index]).convert('RGB')

        imgA = self.tf(imgA)
        imgB = self.tf(imgB)

        return imgA, imgB
    

    def __len__(self):
        return min(len(self.all_imgA), len(self.all_imgB))








