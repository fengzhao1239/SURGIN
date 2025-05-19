from src.ufno import Net3d
from src.utility import OperatorDataset, load_hdf5

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import neptune
from tqdm import tqdm
import os

NEPTUNE_API = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODlkODk2OC1lZTUzLTRkNGItODdmOC0zNDdhNGFmYzU4ZjIifQ=='

class Config:
    def __init__(self):
        self.data_path = '../dataset/Multi_Cartesian.hdf5'
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_save_path = '../checkpoint/ufno.pth'
        self.tags = ['ufno']
        self.step_size = 2
        self.gamma = 0.9
        
        self.mode1 = 10
        self.mode2 = 10
        self.mode3 = 10
        self.width = 36
        


class Trainer:
    def __init__(self, config):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.config = config
        self.device = config.device
        self.model = Net3d(config.mode1, config.mode2, config.mode3, config.width).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)
        self.criterion = nn.MSELoss()
        full_dataset = OperatorDataset(load_hdf5(config.data_path))
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False)
        
        
        self.run = neptune.init_run(
            project = 'DiffNO',
            api_token = NEPTUNE_API,
            tags = config.tags
        )
    
    def train(self):
        for epoch in tqdm(range(self.config.num_epochs)):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                self.run['train/batch_loss'].log(loss.item())
            
            train_loss /= len(self.train_dataloader)
            
            self.run['train/loss'].log(train_loss)
            
            if (epoch + 1) % 10 == 0:
                self.validate(epoch)
                self.save_ckpt(epoch)
            
            self.scheduler.step()
        self.save_ckpt(self.config.num_epochs)
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                val_loss += loss.item()
                
                self.run['val/batch_loss'].log(loss.item())
        
        val_loss /= len(self.val_dataloader)
        self.run['val/loss'].log(val_loss, step=epoch)
    
    def save_ckpt(self, epoch):
        save_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_data, self.config.model_save_path)


if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.train()