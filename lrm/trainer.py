# Copyright (c) 2023, Nico Graf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import random
import math
import os
import imageio
import mcubes
import trimesh
import numpy as np
import numpy.typing as npt
import argparse
import matplotlib.pyplot as plt
from icecream import ic
from typing import Dict
from loss_visualizer import LossVisualizer
from pathlib import Path  
from torch.utils.data import DataLoader
from typing import Tuple, List
from tqdm import tqdm
from .models.training.data import SpikeImageDataset
from .models.training.loss import LRMLoss


from .models.generator import LRMGenerator
from .cam_utils import build_camera_principle, build_camera_standard, center_looking_at_camera_pose


class LRMTrainer:
    def __init__(self, model_name: str):

        # Check if cuda is available to determine device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Compute with GPU")
        else:
            device = torch.device('cpu')
            print("Compute with CPU")

        self.device = device

        _checkpoint = self._load_checkpoint(model_name)
        #ic(_checkpoint.keys())
        #print(f'kwargs: \n{_checkpoint["kwargs"]}\n')
        _model_weights, _model_kwargs = _checkpoint['weights'], _checkpoint['kwargs']['model']
        self.model = self._build_model(_model_kwargs, _model_weights).eval()

        self.infer_kwargs = _checkpoint['kwargs']['infer']
        self.checkpoint = _checkpoint

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _load_checkpoint(self, model_name: str, cache_dir = './.cache'):
        # download checkpoint if not exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cache_dir, f'{model_name}.pth')):
            # os.system(f'wget -O {os.path.join(cache_dir, f"{model_name}.pth")} https://zxhezexin.com/modelzoo/openlrm/{model_name}.pth')
            # raise FileNotFoundError(f"Checkpoint {model_name} not found in {cache_dir}")
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(repo_id='zxhezexin/OpenLRM', filename=f'{model_name}.pth', local_dir=cache_dir)
        else:
            local_path = os.path.join(cache_dir, f'{model_name}.pth')
        checkpoint = torch.load(local_path, map_location=self.device)
        return checkpoint

    def _store_checkpoint(self, checkpoint, model_name: str, dir = './.cache'):
        save_path = Path(dir,f"{model_name}.pth")
        torch.save(checkpoint, save_path)
        
    def _build_model(self, model_kwargs, model_weights):
        model = LRMGenerator(**model_kwargs).to(self.device)
        model.load_state_dict(model_weights)
        print(f"======== Loaded model from checkpoint ========")
        return model

    @staticmethod
    def _get_surrounding_views(thetas : torch.Tensor, radius: float = 2.0, height: float = 0.8) -> torch.Tensor:
        """Compute extrinisic camera settings for given radii

        Args:
            thetas (Tensor): Radiant angles for camera (batch_size).
            radius (float, optional): camera radius. Defaults to 2.0.
            height (float, optional): camera position height. Defaults to 0.8.

        Returns:
            torch.tensor: camera extrinsics
        """
        # angles
        # radius: camera dist to center
        # height: height of the camera
        # return: (M, 3, 4)
        assert radius > 0

        camera_positions = []
        projected_radius = torch.sqrt(torch.tensor(radius ** 2 - height ** 2))
        x = projected_radius * torch.cos(thetas)
        y = projected_radius * torch.sin(thetas)
        z = height * torch.ones_like(thetas)
        camera_positions = torch.stack([x,y,z], dim=1) # (batch_size, 3)
        extrinsics = center_looking_at_camera_pose(camera_positions)

        return extrinsics

    @staticmethod
    def _default_intrinsics():
        # return: (3, 2)
        fx = fy = 384
        cx = cy = 256
        w = h = 512
        intrinsics = torch.tensor([
            [fx, fy],
            [cx, cy],
            [w, h],
        ], dtype=torch.float32)
        return intrinsics

    def _default_source_camera(self, batch_size: int = 1) -> torch.Tensor:
        """Get source camera position

        The same setting will be returned for every element in the batch

        Args:
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            torch.Tensor : (batch_size, camera extrinsics)
        """
        dist_to_center = 2.0
        # images live in the (x, y) plane, where the y-plane is upside down
        # we want negative image y direction to be z direction.
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center], 
            [0, 1, 0, 0],
        ]], dtype=torch.float32)
        canonical_camera_intrinsics = self._default_intrinsics().unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, thetas : npt.NDArray) -> torch.Tensor:
        """Compute render cameras 

        The camera intrinsics are taken from the default intrinsics.
        The extrinsics are computed for the default number of extrinisc
        views.

        Args:
            thetas (NDArray): Camera angles in radiant (circular around image) size (batch_size, #views).

        Returns:
            torch.Tensor: (batch_size, #camera_angles, intrinsics and extrinsics)
        """
        thetas_shape = thetas.shape
        radius=2
        # default height is 0.8, but for training we use 0
        height=0
        
        # collapse dimensions
        thetas = thetas.flatten()
        
        # extrinsics determine camera position
        render_camera_extrinsics = self._get_surrounding_views(thetas=thetas, radius=radius, height=height)

        # intrinsics determine internal camera setting for all batches and views (dimension 0 is the dimension for views)
        render_camera_intrinsics = self._default_intrinsics().unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)

        # get rendrer cameras as vectors 25 = (16 + 9) for each angle : #num_views
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
        
        # restore collapsed batchsize and views
        render_cameras = torch.unflatten(render_cameras, dim=0, sizes=thetas_shape)
        
        return render_cameras

    def infer_batch(self, images: torch.Tensor, thetas : npt.NDArray, render_size: int) -> torch.Tensor:
        # image: [batch_size, C_img, H_img, W_img]

        chunk_size = 2
        assert thetas.shape[1] % chunk_size == 0, "Number of angles are not divisible by chunk size"
        batch_size = images.shape[0]
        
        source_camera = self._default_source_camera(batch_size).to(self.device)
        render_cameras = self._default_render_cameras(thetas=thetas).to(self.device)

        planes = self.model.forward_planes(images.to(self.device), source_camera)

        # forward synthesizer per mini-batch
        frames = []
        for i in range(0, render_cameras.shape[1], chunk_size):
            frames.append(
                self.model.synthesizer(
                    planes,
                    render_cameras[:, i:i+chunk_size],
                    render_size,
                )
            )
        # merge rgb frames
        frames = torch.cat([r['images_rgb'] for r in frames], dim=1)

        return frames
    
    def train(self, train_path : str, source_size: int):

        source_image_size = int(source_size if source_size > 0 else self.infer_kwargs['source_size'])
        
        # Fix seeds to make all the code reproducible
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        # specify some training parameters
        num_epochs = 3
        num_worker = 8 # number of threads / cores used
        batch_size = 1
        V = 2
        lam = 2
        
        # Define trainig algorithms
        loss_function = LRMLoss(lam=lam)
        dataset = SpikeImageDataset(image_base_dir=train_path, V=V, source_size=source_image_size, cache_imgs=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-4)
        
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer)
        scheduler = None
        
        model, train_log = self.train_model(
                            train_data=dataset,
                            optimizer=optimizer,
                            loss_function=loss_function,
                            scheduler=scheduler,
                            model=self.model,
                            num_epochs=num_epochs,
                            num_workers=num_worker,
                            batch_size=batch_size)

        LossVisualizer().plot_loss(train_loss=train_log, val_loss=None, num_epochs=num_epochs)

    def train_model(self,
                    train_data : SpikeImageDataset ,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    loss_function: torch.nn.Module,
                    scheduler: torch.optim.lr_scheduler,
                    num_epochs: int,
                    num_workers: int = 8,
                    batch_size: int = 8,
                    device: torch.device = torch.device('cpu')) -> Tuple[torch.nn.Module, List[float], List[float]]:

        # set model to training mode
        model.train()
        model.to(device)

        # get data loaders
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # store losses for logging
        train_loss_log = []

        train_loss = []
        lpips_loss = []
        mse_loss = []
        # training loop (tqdm is just for nice utput but deactivated)
        for epoch in tqdm(range(num_epochs), disable=True):
            # Training
            model.train()
            for i, (source_imgs, train_imgs, thetas) in enumerate(train_loader):
               
                ic('Batch ',i)
                ic(thetas)

                optimizer.zero_grad()
                outputs = self.infer_batch(images=source_imgs, thetas=thetas, render_size=train_data.source_size)
                outputs = outputs.clamp(0,1)
                
                # compute loss
                loss, _lpips_loss, _mse_loss = loss_function(outputs, train_imgs)

                # update weights
                loss.backward()
                optimizer.step()

                if not scheduler is None:
                    scheduler.step()

                train_loss.append(loss.item())
                mse_loss.append(_mse_loss.item())
                lpips_loss.append(_lpips_loss.item())
                

            ic(epoch, "Mean Train Loss so far:", np.mean(train_loss))
            train_loss = np.array(train_loss)
            mse_loss = np.array(mse_loss)
            lpips_loss = np.array(lpips_loss)
            loss = np.stack([train_loss, mse_loss, lpips_loss])
            np.save("./cache/losses.npy")

        return model, train_loss, mse_loss, lpips_loss
        

if __name__ == '__main__':

    """
    Example usage:
    python -m lrm.inferrer --model_name lrm-base-obj-v1 --source_image ./assets/sample_input/owl.png --export_video --export_mesh
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='lrm-base-obj-v1')
    parser.add_argument('--train_path', type=str, default='./assets/train_data')
    parser.add_argument('--output_dir', type=str, default='./dump/output_data', help='Folder to store several outputs including trained model.')
    parser.add_argument('--source_size', type=int, default=-1)
    args = parser.parse_args()

    train_data_path = Path(args.train_path)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with LRMTrainer(model_name=args.model_name) as trainer:
        model, train_loss, mse_loss, lpips_loss = trainer.train(train_path=train_data_path, source_size=args.source_size)
        checkpoint = trainer.checkpoint
        checkpoint['weights'] = model.state_dict() # store model
        trainer._store_checkpoint(checkpoint=checkpoint, model_name="trained_model")

        
