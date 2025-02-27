# Copyright 2023 Antoine Schnepf, Karim Kassab

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import wandb 
import torchvision.transforms as tr
import numpy as np
import sys
import random
import string
import tqdm 
from ae.utils import make_dashboard_video
from ae.utils import AverageMeter


class Evaluator:

    def __init__(self, train_scenes, test_scenes, pose_sampler, config, repo_path) : 
        self.train_scenes = train_scenes
        self.test_scenes = test_scenes
        self.pose_sampler = pose_sampler
        #self.lpips_loss = get_lpips_loss(device) #TODO: add LPIPS loss ? 
        self.config = config
        self.repo_path = repo_path
    
    @staticmethod
    def compute_mse_psnr(im1:torch.Tensor, im2:torch.Tensor, max2=4):
        "Compute the PSNR between two tensor images, expected with shape [... C, H, W] and values in [-1, 1] "

        mse = torch.nn.functional.mse_loss(im1, im2)
        psnr = 10 * torch.log10(max2 / mse)

        return dict(
            psnr=psnr.item(),
            mse=mse.item()
        )
    
    @torch.no_grad()
    def _get_lnerf_metrics(self, vae, latent_nerf, renderer, single_scene_data, normalizer, batch_size, device):
            
            latent_nerf = latent_nerf.to(device)
            
            latent_psnr_meter = AverageMeter()  
            latent_mse_meter = AverageMeter()
            rgb_psnr_meter = AverageMeter()  
            rgb_mse_meter = AverageMeter()
            vae_psnr_meter = AverageMeter()  
            vae_mse_meter = AverageMeter()
            low_res_rgb_psnr_meter = AverageMeter()  
            low_res_rgb_mse_meter = AverageMeter()


            loader = torch.utils.data.DataLoader(single_scene_data, batch_size=batch_size, shuffle=False)

            for i, batch in enumerate(loader) :
                # getting g.t. image and pose
                rgb_img = batch['img'].to(device)
                pose = batch['pose'].to(device).unsqueeze(0)

                # getting pseudo g.t. latent img
                latent_img = vae.encode(rgb_img).latent_dist.sample()
                normalized_latent_img = normalizer.normalize(latent_img)

                # getting reconstructed latent img 
                rendering = renderer(latent_nerf.unsqueeze(0), pose)
                normalized_latent_img_render = rendering['img'].squeeze(0)
                latent_img_render = normalizer.denormalize(normalized_latent_img_render)
                if self.config.latent_nerf.concat_low_res_rgb: 
                    low_res_rgb_img = torch.nn.functional.interpolate(rgb_img, size=(self.config.latent_nerf.img_size, self.config.latent_nerf.img_size), mode='bilinear')
                    low_res_rgb_img_render = rendering['img_concat'].squeeze(0)

                # getting reconstruced rgb img
                rgb_img_render = vae.decode(latent_img_render).sample

                # getting autoencoded rgb img
                rgb_img_ae = vae.decode(latent_img).sample
                
                lat_psnr, lat_mse = self.compute_mse_psnr(normalized_latent_img, normalized_latent_img_render).values()
                rgb_psnr, rgb_mse = self.compute_mse_psnr(rgb_img, rgb_img_render).values()
                vae_psnr, vae_mse = self.compute_mse_psnr(rgb_img, rgb_img_ae).values()
                if self.config.latent_nerf.concat_low_res_rgb: 
                    low_res_rgb_psnr, low_res_rgb_mse = self.compute_mse_psnr(low_res_rgb_img, low_res_rgb_img_render).values()


                latent_psnr_meter.update(lat_psnr)
                latent_mse_meter.update(lat_mse)
                rgb_psnr_meter.update(rgb_psnr)
                rgb_mse_meter.update(rgb_mse)
                vae_psnr_meter.update(vae_psnr)
                vae_mse_meter.update(vae_mse)
                if self.config.latent_nerf.concat_low_res_rgb: 
                    low_res_rgb_psnr_meter.update(low_res_rgb_psnr)
                    low_res_rgb_mse_meter.update(low_res_rgb_mse)


            metrics = {
                'latent_nerf' : {
                    'psnr' : latent_psnr_meter.avg,
                    #'mse' : latent_mse_meter.avg
                },
                'rgb_nerf' : {
                    'psnr' : rgb_psnr_meter.avg,
                    #'mse' : rgb_mse_meter.avg
                },
                'vae' : {
                    'psnr' : vae_psnr_meter.avg,
                    #'mse' : vae_mse_meter.avg
                },
                'low_res_rgb' : {
                    'psnr' : low_res_rgb_psnr_meter.avg,
                    #'mse' : low_res_rgb_mse_meter.avg
                }
            }

            return metrics

    def _get_rgb_override_metrics(self, vae, nerf, renderer, single_scene_data, normalizer, batch_size, device):
        nerf = nerf.to(device)

        rgb_psnr_meter = AverageMeter()  
        rgb_mse_meter = AverageMeter()

        loader = torch.utils.data.DataLoader(single_scene_data, batch_size=batch_size, shuffle=False)

        for i, batch in enumerate(loader) :
            # getting g.t. image and pose
            rgb_img = batch['img'].to(device)
            pose = batch['pose'].to(device).unsqueeze(0)

            # getting reconstructed latent img 
            rendering = renderer(nerf.unsqueeze(0), pose)
            img_render = rendering['img'].squeeze(0)
            
            rgb_psnr, rgb_mse = self.compute_mse_psnr(rgb_img, img_render).values()

            rgb_psnr_meter.update(rgb_psnr)
            rgb_mse_meter.update(rgb_mse)


        metrics = {
            'rgb_override' : {
                'psnr' : rgb_psnr_meter.avg,
                #'mse' : rgb_mse_meter.avg
            }
        }

        return metrics
    @torch.no_grad()
    def get_evaluation_metrics(self, vae, latent_nerfs, renderer, normalizer, batch_size, device):
        
        metrics = {}
        if self.config.rgb_override:
            metrics_fn = self._get_rgb_override_metrics
        else: 
            metrics_fn = self._get_lnerf_metrics

        for scene_idx in tqdm.tqdm(range(len(latent_nerfs)), desc="Computing evaluation metrics") :
            metrics[f'scene_{scene_idx}'] = {
                'train_views' : metrics_fn(vae, latent_nerfs[scene_idx], renderer, self.train_scenes[scene_idx], normalizer, batch_size, device),
                'test_views' : metrics_fn(vae, latent_nerfs[scene_idx], renderer, self.test_scenes[scene_idx], normalizer, batch_size, device)
            }
        
        mean_metric_dict = {}
        for split in list(metrics.values())[0].keys() :
            mean_metric_dict[split] = {}
            for model in metrics['scene_0'][split].keys() :
                mean_metric_dict[split][model] = {}
                for metric_name in metrics['scene_0'][split][model].keys() :
                    mean_metric_dict[split][model][metric_name] = np.mean([
                        metrics[f'scene_{scene_idx}'][split][model][metric_name] for scene_idx in range(len(latent_nerfs))
                    ])

        if not self.config.eval.metrics.log_scenes_independently : 
            return mean_metric_dict
        
        metrics['all_scenes'] = mean_metric_dict
        
        return metrics
    
    @torch.no_grad()
    def get_video_eval(self, vae, latent_nerfs, renderer, normalizer, batch_size, device):
        """Makes a video of the given latent nerfs
        Returns the wandb.Video object"""

        vae.to(device)
        
        latent_to_pil_transform = tr.Compose([
            tr.Normalize(-1, 2),
            tr.Lambda(lambda x : x.clamp(0, 1)),
            #tr.Lambda(lambda x : x[:3]),
            tr.ToPILImage()
        ])

        if self.config.rgb_override:
            latent_to_rgb_transform = latent_to_pil_transform

        else: 
            def _decode_latent(latent_img):
                latent_img.to(device)
                latent_img = normalizer.denormalize(latent_img.unsqueeze(0))
                rgb_img = vae.decode(latent_img).sample.squeeze(0)
                return rgb_img
            
            
            latent_to_rgb_transform = tr.Compose([
                tr.Lambda(_decode_latent),
                tr.Lambda(lambda x: x.clamp(-1,1)),
                tr.Lambda(lambda x: (x + 1)/2),
                tr.ToPILImage()
            ])

            
        rd_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        video_savename = f"latent_nerf_video_{rd_string}.mp4"
        make_dashboard_video(
            renderer, 
            latent_nerfs, 
            self.pose_sampler, 
            latent_to_pil_transform, 
            latent_to_rgb_transform, 
            self.config.latent_nerf.rendering_options, 
            batch_size, 
            device,
            video_savename, 
            save_dir=os.path.join(self.repo_path, self.config.savedir, 'buffer'), 
            n_frames=self.config.eval.video.n_frames,
            fps=self.config.eval.video.fps,
            azimuth_range=self.config.eval.video.azimuth_range,
            elevation_range=self.config.eval.video.elevation_range,
            radius_range=self.config.eval.video.radius_range
        )
        
        return wandb.Video(os.path.join(self.repo_path, self.config.savedir, 'buffer', video_savename))
        