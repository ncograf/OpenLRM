import lpips
import torch
import torch.nn as nn
import numpy as np
import time
from icecream import ic
import matplotlib.pyplot as plt

class LRMLoss(nn.Module):
    
    def __init__(self, lam : float):

        super(LRMLoss, self).__init__()
        self.lam = lam

        # get mse loss for each image
        self.mse_loss = nn.MSELoss()
        
        # first compute mse loss
        # get lpips function image values need to be normalized to [-1,1]!!!!
        self.lpips_loss = lpips.LPIPS(net='alex')
    
    def scale_rgb(self, image, factor : float = 0.5, cent : float = 1.):
        """Function to scale rgb images to the desired range
        
        With the right arguments, scales rgb image to [-1,1]

        Source:
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/__init__.py#L86

        Args:
            image (array_like): _description_
            factor (float, optional): factor to devide by. Defaults to 0.5.
            cent (float, optional): center to be subtracted. Defaults to 1..

        Returns:
            torch.Tensor: scaled image
        """
        return torch.Tensor((image / factor - cent))

    def forward(self, output : torch.Tensor, target : torch.Tensor):
        """Computes loss for one object

        Given V images for each elemnt in the batch

        Args:
            output (torch.Tensor): inferred images (batch_size, V, 3, img_width, img_height)
            target (torch.Tensor): target images (batch_size, V, 3, img_width, img_height)
        """
        
        assert output.shape == target.shape
        
        V = output.shape[1] # number of images per object to compare

        total_loss = 0.0
        total_mse = 0.0
        total_lpips = 0.0

        for i in range(V):
            tmp_output = output[:,i,:,:]
            tmp_target = target[:,i,:,:]
            
            tmp_mse = self.mse_loss(tmp_output, tmp_target)

            tmp_output = self.scale_rgb(tmp_output)
            tmp_target = self.scale_rgb(tmp_target)

            tmp_lpips = self.lpips_loss(tmp_output, tmp_target)
            current_loss = tmp_mse + self.lam * tmp_lpips
            total_loss += current_loss
            total_mse += tmp_mse
            total_lpips += tmp_lpips
        
        total_loss = total_loss / V

        return total_loss, total_lpips, total_mse

