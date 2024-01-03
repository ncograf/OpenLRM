from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from icecream import ic
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

class SpikeImageDataset(Dataset):
    
    def __init__(self, image_base_dir : Path, V : int, source_size : int, cache_imgs : bool = True, dtype : torch.dtype = torch.float32):

        # In the base directory every spike has one folder
        # In the spikes folder there are images around the spike
        # computed with the same angle in between the spikes
        self.image_base_dir : Path = image_base_dir
        
        self.source_size : int = source_size
        self.cache_imgs : bool = cache_imgs
        self.dtype = dtype
        
        # Number of images to be considered for each spike
        self.V = V
        
        # dictionnary with paths to the V images
        self.image_dict : Dict[str, List[Tuple[str, float]]] = {}
        self.images : Dict[str, torch.Tensor] = {}
        
        self.draw_image_views() 
        

    def draw_image_views(self, seed = None):
        if seed is None:
            generator = np.random.default_rng()
        else:
            generator = np.random.default_rng(seed)
        
        # reset dictonaries
        self.image_dict = {}
        self.images = {}
        for img_dir in self.image_base_dir.iterdir():
            if img_dir.is_dir():
                # get images
                images = np.array(sorted([*img_dir.iterdir()], key=lambda f: int(f.stem)))
                n_images = len(images)

                # get the corresponding phis by convention
                phis = np.linspace(start = 0,stop=2 * np.pi, num=n_images, endpoint=False)
                
                # randomly choose training files
                choice = generator.choice(n_images, size=self.V, replace=False)
                images = images[choice]
                phis = phis[choice]
                
                # adjust angle to be relative to the first image
                phis = phis - phis[0]
                
                # store on RAM
                self.image_dict[img_dir.name] = zip(images.tolist(), phis)

    def __len__(self):
        return len(self.image_dict)
    
    def __getitem__(self, idx):
        """Returns a tensor (V, 3, W, H)
        
        Args:
            idx (int): Index of chosen image
        """
        
        # Read images if not already storred
        name = list(self.image_dict.keys())[idx]
        if name in self.images.keys():
            ic("cached images")
            phis = [phi for _, phi in self.image_dict[name]]
            images = self.images[name]
        else:
            img_list = []
            phis = []
            for file, phi in self.image_dict[name]:
                img = read_image(str(file))

                # images in image folder need to have right dimension
                assert img.shape[1] == self.source_size
                assert img.shape[2] == self.source_size

                img_list.append(img)
                phis.append(phi)

            phis = torch.tensor(phis, dtype=self.dtype)
            images =  torch.stack(img_list, dim=0).type(self.dtype) / 255
            images = images.clamp(0,1)
            
            if self.cache_imgs:
                self.images[name] = images

        return images[0], images, phis
        
