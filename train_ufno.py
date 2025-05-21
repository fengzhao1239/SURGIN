from src.ufno import Net3d
from src.utility import OperatorDataset, load_hdf5
from src.lploss import LpLoss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import neptune
from tqdm import tqdm
import os

NEPTUNE_API = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ODlkODk2OC1lZTUzLTRkNGItODdmOC0zNDdhNGFmYzU4ZjIifQ=='

class Config:
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        self.data_path = 'dataset/Multi_Cartesian.hdf5'
        self.batch_size = 50
        self.num_epochs = 150
        self.learning_rate = 0.001
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_save_path = 'checkpoint/ufno_sat.pth'
        self.tags = ['ufno']
        self.step_size = 2
        self.gamma = 0.9
        self.corf = 0.5
        
        self.mode1 = 10
        self.mode2 = 10
        self.mode3 = 8
        self.width = 36
        
        torch.manual_seed(42)
        


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = Net3d(config.mode1, config.mode2, config.mode3, config.width).to(self.device)
        print(f"----- Model parameters: {self.model.count_params()}")
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)
        self.criterion = LpLoss(d=2, p=2, size_average=True, reduction=True)
        full_dataset = OperatorDataset(load_hdf5(config.data_path), ps_flag='sat')
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
    
    def derivative_loss(self, pred, label, loss_fn):
        assert pred.shape == label.shape
        assert len(pred.shape) == 4, f"Expected 4D tensor, got {pred.shape} tensor"
        pred_dx = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        label_dx = label[:, 1:, :, :] - label[:, :-1, :, :]
        label_dy = label[:, :, 1:, :] - label[:, :, :-1, :]
        loss_dx = loss_fn(pred_dx, label_dx)
        loss_dy = loss_fn(pred_dy, label_dy)
        return loss_dx + loss_dy
    
    def train(self):
        for epoch in tqdm(range(self.config.num_epochs)):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y) + self.derivative_loss(output, y, self.criterion) * config.corf
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                self.run['train/batch_loss'].append(loss.item())
            
            train_loss /= len(self.train_dataloader)
            
            self.run['train/loss'].append(train_loss)
            self.plot(x, y, output)
            
            if (epoch + 1) % 10 == 0:
                self.validate(epoch)
                # self.save_ckpt(epoch)
            
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
                
                self.run['val/batch_loss'].append(loss.item())
        
        val_loss /= len(self.val_dataloader)
        self.run['val/loss'].append(val_loss, step=epoch)
    
    def save_ckpt(self, epoch):
        save_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_data, self.config.model_save_path)

    def plot(self, x, y, output):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        im0 = ax[0].imshow(x[0, :, :, -1, 0].detach().cpu().numpy(), cmap='jet')
        ax[0].set_title('Input')
        fig.colorbar(im0, ax=ax[0], shrink=0.5)
        im1 = ax[1].imshow(y[0, :, :, -1].detach().cpu().numpy(), cmap='jet')
        ax[1].set_title('Target')
        fig.colorbar(im1, ax=ax[1], shrink=0.5)
        im2 = ax[2].imshow(output[0, :, :, -1].detach().cpu().numpy(), cmap='jet')
        ax[2].set_title('Output')
        fig.colorbar(im2, ax=ax[2], shrink=0.5)
        self.run['train/plot'].append(fig)
        plt.close(fig)


if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.train()