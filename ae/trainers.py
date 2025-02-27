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

import os, sys
import random

import torch
import wandb
import time
import pickle
import itertools
import tqdm
import copy
from diffusers import AutoencoderKL

from ae.utils import AverageMeter, compute_tv, save_config
from ae.triplane_renderer import TriPlaneRenderer
from ae.camera_utils import LazyPoseSampler
from ae.utils import do_now, set_requires_grad, to_dict, compute_kl_div
from datasets.multiview_dataset import DatasetBank, ObjectDataset, MultiSceneDataset, CachedDataset
from ae.global_triplanes import TriplaneManager



def init_datasets(config) : 
    mv_dataset = DatasetBank.instantiate_dataset(
        img_size=config.rgb_img_size,
        dataset_name=config.dataset_name,
        split='train',
        n_views='max',
        max_obj=config.scenes_range_start + config.n_scenes
    )
    pose_sampler = LazyPoseSampler(
        dataset_name=config.dataset_name,
    )
    train_scenes = [ 
        ObjectDataset(mv_dataset, selected_obj_idx=i, split='train')
        for i in range(config.scenes_range_start, config.scenes_range_start + config.n_scenes)
    ]
    multi_scene_trainset = MultiSceneDataset(train_scenes)
    test_scenes = [ 
        ObjectDataset(mv_dataset, selected_obj_idx=i, split='test')
        for i in range(config.scenes_range_start, config.scenes_range_start + config.n_scenes)
    ]
    multi_scene_testset = MultiSceneDataset(test_scenes)
    return train_scenes, multi_scene_trainset, test_scenes, multi_scene_testset, pose_sampler

def init_models(config, device) : 

    if config.rgb_override:
        return init_models_rgb_override(config, device)
    
    # 1. Vae
    vae = AutoencoderKL.from_pretrained(
        config.vae.pretrained_model_name_or_path, 
        subfolder="vae", revision=config.vae.revision
    )

    # Handlong RGB Low Res option: n_channels and bg_color update
    config.latent_nerf.rendering_options['bg_color'] = config.vae.normalization.img.bg_color
    if config.latent_nerf.concat_low_res_rgb : 
        config.latent_nerf.rendering_options['bg_color'] += config.vae.normalization.img.rgb_bg_color


    n_features = config.latent_nerf.n_local_features
    if config.global_latent_nerf.apply:
        if config.global_latent_nerf.fusion_mode == "concat":
            n_features = config.global_latent_nerf.n_global_features + config.global_latent_nerf.n_local_features
        elif config.global_latent_nerf.fusion_mode == "sum":
            n_features = config.global_latent_nerf.n_global_features

    latent_renderer = TriPlaneRenderer(
        neural_rendering_resolution=config.latent_nerf.img_size,
        n_main_channels=config.latent_nerf.n_latent_channels,
        n_concat_channels=config.latent_nerf.n_concat_channels,
        do_concatenate=config.latent_nerf.concat_low_res_rgb,
        n_features=n_features,
        aggregation_mode=config.latent_nerf.aggregation_mode,
        rendering_kwargs=config.latent_nerf.rendering_options,
        use_coordinates=False,
        use_directions=False
    )
    latent_nerfs =  TriplaneManager(
            n_scenes=config.n_scenes,
            local_nerf_cfg=config.latent_nerf,
            global_nerf_cfg=config.global_latent_nerf,
            device_=device
    )
    
    return vae, latent_renderer, latent_nerfs

def init_models_rgb_override(config, device) :

    class EncoderOutput:
        def __init__(self, mean):
            self.mean = mean
            self.latent_dist = IdentityDistribution(mean)

    class DecoderOutput():
        def __init__(self, sample):
            self.sample = sample

    class IdentityDistribution():
        def __init__(self, x):
            self.mean = x
            self.std = torch.zeros_like(x)

        def sample(self):
            return self.mean

    class IdentityEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return EncoderOutput(x)
    
    class IdentityDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return DecoderOutput(x)

    class IdentityVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = IdentityEncoder()
            self.decoder = IdentityDecoder()

        def encode(self, x):
            return self.encoder(x)
        
        def decode(self, z):
            return self.decoder(z)
        
        def forward(self, x):
            return self.decoder(self.encoder(x))
    
    vae = IdentityVAE()

    n_features = config.rgb_nerf_as_latent_override.n_local_features
    if config.global_latent_nerf.apply:
        if config.global_latent_nerf.fusion_mode == "concat":
            n_features = config.global_latent_nerf.n_global_features + config.global_latent_nerf.n_local_features
        elif config.global_latent_nerf.fusion_mode == "sum":
            n_features = config.global_latent_nerf.n_global_features

    latent_renderer = TriPlaneRenderer(
        neural_rendering_resolution=config.rgb_nerf_as_latent_override.img_size,
        n_main_channels=config.rgb_nerf_as_latent_override.n_channels,
        n_concat_channels=0,
        do_concatenate=False,
        n_features=n_features,
        aggregation_mode=config.rgb_nerf_as_latent_override.aggregation_mode,
        rendering_kwargs=config.rgb_nerf_as_latent_override.rendering_options,
        use_coordinates=False,
        use_directions=False
    )
    latent_nerfs =  TriplaneManager(
            n_scenes=config.n_scenes,
            local_nerf_cfg=config.rgb_nerf_as_latent_override,
            global_nerf_cfg=config.global_latent_nerf,
            device_=device
    )
    return vae, latent_renderer, latent_nerfs


def warmup(dataloader, vae, normalizer, device):

    def warmup_iter(batch): 
        x = batch['img'].to(device)
        latent_dist = vae.encode(x).latent_dist
        z = latent_dist.sample()
        normalizer.pop_push(z.detach())
    
    for batch in itertools.cycle(dataloader) :
        if normalizer.is_full() :
            print("Warming up the sliding window: DONE")
            break
        warmup_iter(batch)

def train(config, 
          t_args, 
          expname, 
          multi_scene_set, 
          vae, 
          latent_renderer, 
          latent_nerfs, 
          normalizer, 
          criterion, 
          evaluator, 
          device, 
          repo_path,
          debug=False): 
        
        use_encoder = t_args.losses.loss_lnerf or t_args.losses.loss_ae
        use_decoder = t_args.losses.loss_alpha_d or t_args.losses.loss_ae

        # Defining dataloader
        if use_encoder and t_args.cache_latents.apply: 
            assert t_args.freezes.freeze_encoder, "Caching latents should only be used with a frozen encoder"
            multi_scene_set = CachedDataset(multi_scene_set, vae, device, t_args.cache_latents.batch_size, t_args.cache_latents.use_mean, repo_path, config.savedir)
        dataloader = torch.utils.data.DataLoader(multi_scene_set, batch_size=t_args.batch_size, shuffle=True)

        # Warming up sliding window
        if use_encoder and t_args.warmup_window:
            warmup(dataloader, vae, normalizer, device)

        # Sanity check
        assert normalizer.is_full(), "The sliding window is not full" 

        # Defining latent nerf optimizer and scheduler
        if t_args.scheduler.type == 'multistep': 
            Scheduler = torch.optim.lr_scheduler.MultiStepLR
            scheduler_kwargs = t_args.scheduler.multistep_config
        elif t_args.scheduler.type  == 'exp':
            Scheduler = torch.optim.lr_scheduler.ExponentialLR
            scheduler_kwargs = t_args.scheduler.exp_config
        else: 
            raise ValueError(f"Scheduler type {t_args.scheduler.type} not recognized")

        optimizer_latent_nerfs = torch.optim.Adam([
            {'params' : latent_renderer.parameters(), 'lr' : t_args.latent_nerf.tinymlp_lr},
            {'params' : latent_nerfs.local_planes,                 'lr' : t_args.latent_nerf.lr},
        ])
        scheduler_latent_nerfs = Scheduler(
            optimizer_latent_nerfs, 
            **scheduler_kwargs
        )

        if config.global_latent_nerf.apply :
            # Defining global latent nerf optimizer and scheduler
            optimizer_global_latent_nerfs = torch.optim.Adam([{
                'params' : latent_nerfs.global_planes,                 
                'lr' : t_args.global_latent_nerf.lr
            }])
            scheduler_global_latent_nerfs = Scheduler(
                optimizer_global_latent_nerfs, 
                **scheduler_kwargs
            )
            # Defining base coefs optimizer and scheduler
            optimizer_base_coefs = torch.optim.Adam([{
                'params' : latent_nerfs.coefs,                 
                'lr' : t_args.base_coefs.lr
            }])
            scheduler_base_coefs = Scheduler(
                optimizer_base_coefs, 
                **scheduler_kwargs
            )

        # Defining encoder optimizer and scheduler
        if use_encoder:
            optimizer_encoder = torch.optim.Adam([{
                'params' : vae.encoder.parameters(), 
                'lr' : t_args.encoder.lr
            }])
            scheduler_encoder = Scheduler(
                optimizer_encoder, 
                **scheduler_kwargs
            ) 
            #vae.encoder.to(device)

        # Defining decoder optimizer and scheduler
        if use_decoder:
            optimizer_decoder = torch.optim.Adam([{
                'params' : vae.decoder.parameters(), 
                'lr' : t_args.decoder.lr
            }])
            scheduler_decoder = Scheduler(
                optimizer_decoder, 
                **scheduler_kwargs
            )
            #vae.decoder.to(device)

        # Logging init
        wandb.init(
            entity='criteo-3d',
            project=config.wandb_project_name,
            config=copy.deepcopy(config),
            dir=repo_path+"/wandb",
            notes=t_args.wandb_note
        )
        loss_ae_meter = AverageMeter()
        loss_alpha_D_meter = AverageMeter()
        loss_latent_nerf_meter = AverageMeter()
        loss_alpha_D_meter = AverageMeter()
        loss_rgb_low_res_meter = AverageMeter()
        loss_rgb_override_meter = AverageMeter()
        tv_depth_meter = AverageMeter()
        tv_meter = AverageMeter()
        kl_div_meter = AverageMeter()
        loss_meter = AverageMeter()
            
        # Freezing models if necessary
        set_requires_grad(vae.encoder, not t_args.freezes.freeze_encoder)
        set_requires_grad(vae.decoder, not t_args.freezes.freeze_decoder)

        latent_nerfs.device_ = device

        def train_iter(batch): 
            train_iter_time_start = time.time()
            # 1. Get data
            rgb_img = batch['img'].to(device)
            pose = batch['pose'].unsqueeze(1).to(device) #shape [bs, n_renders(=1), 25]
            scene_idxs = batch['scene_idx']
            latent_nerf = latent_nerfs[scene_idxs].squeeze(1)

            if use_encoder:
                #2. Encode and decode input img
                if t_args.cache_latents.apply:
                    latent_img = batch['cached_latent'].to(device)
                else : 
                    latent_dist = vae.encode(rgb_img).latent_dist
                    if config.latent_nerf.mu_nerf:
                        latent_img = latent_dist.mean
                    else : 
                        latent_img = latent_dist.sample()
                    #latent_img_mean = latent_dist.mean # conflict with cached latents
                    #latent_img_std = latent_dist.std # conflict with cached latents

                #3. Normalize latent img z (via sliding window statistics)
                if config.vae.normalization.img.actualize_window:
                    normalizer.pop_push(latent_img.detach())
                normalized_latent_img = normalizer.normalize(latent_img) 

                #4. Log running mean and std of the sliding window
                if do_now(iter_nb, t_args.log_sw_stats_every_iter):
                    if hasattr(normalizer.agg_tensor, 'running_mean'):
                        wandb.log({
                            'sw_running_mean': to_dict(normalizer.agg_tensor.running_mean),
                            'sw_running_std': to_dict(normalizer.agg_tensor.running_std),
                            'cum_iter': epoch*len(dataloader) + iter_nb
                        })

                if use_decoder:
                    rgb_img_ae = vae.decode(latent_img).sample

            #4. Latent nerf rendering
            rendering = latent_renderer(latent_nerf, pose)
            normalized_latent_img_render = rendering['img'].squeeze(1)

            if config.latent_nerf.concat_low_res_rgb: 
                low_res_rgb_img = torch.nn.functional.interpolate(rgb_img, size=(config.latent_nerf.img_size, config.latent_nerf.img_size), mode='bilinear')
                low_res_rgb_img_render = rendering['img_concat'].squeeze(1)

            latent_depth = rendering['img_depth'].squeeze(1)

            #5. Decode latent rendering into rgb domain
            if use_decoder:
                latent_img_render = normalizer.denormalize(normalized_latent_img_render)
                rgb_img_render = vae.decode(latent_img_render).sample

            #6. Compute losses and optimize
            loss = 0
            if t_args.losses.loss_ae:
                loss_ae = criterion(rgb_img_ae, rgb_img)
                loss_ae_meter.update(loss_ae.item())
                loss += t_args.lambda_ae * loss_ae

            if t_args.losses.loss_lnerf:
                loss_latent_nerf = criterion(normalized_latent_img, normalized_latent_img_render)
                loss_latent_nerf_meter.update(loss_latent_nerf.item())
                loss += t_args.lambda_latent_nerf * loss_latent_nerf

            if t_args.losses.loss_low_res_rgb and config.latent_nerf.concat_low_res_rgb:
                loss_low_res_rgb = criterion(low_res_rgb_img_render, low_res_rgb_img)
                loss_rgb_low_res_meter.update(loss_low_res_rgb.item())
                loss += t_args.lambda_low_res_rgb * loss_low_res_rgb

            if t_args.losses.loss_alpha_d:
                loss_alpha_D = criterion(rgb_img_render, rgb_img)
                loss_alpha_D_meter.update(loss_alpha_D.item())
                loss += t_args.lambda_alpha_D * loss_alpha_D

            if t_args.losses.loss_rgb_override:
                loss_rgb_override = criterion(normalized_latent_img_render, rgb_img)
                loss_rgb_override_meter.update(loss_rgb_override.item())
                loss += t_args.lambda_rgb_override * loss_rgb_override
                
            #6.1 Add eventual regularisation terms
            # KL divergence for VAE
            if t_args.lambda_kl > 0:
                raise NotImplementedError("KL divergence has been un-implemented on 18/02")
                kl_div = compute_kl_div(latent_img_mean, latent_img_std)
                kl_div_meter.update(kl_div.item())
                loss += t_args.lambda_kl * kl_div
            
            # TV regularization for Tri-Planes features
            if t_args.lambda_tv > 0: 
                tv = compute_tv(latent_nerf, mode=t_args.tv_mode) 
                tv_meter.update(tv.item())
                loss += t_args.lambda_tv * tv


            # TV regularization for depths    
            if t_args.lambda_depth_tv > 0:
                depth_tv = compute_tv(latent_depth, mode=t_args.depth_tv_mode) #depth regularisation
                tv_depth_meter.update(depth_tv.item())
                loss += t_args.lambda_depth_tv * depth_tv

            #7. Optimize vae & latent nerf
            loss_meter.update(loss.item())
            loss.backward()

            # for debuging
            if latent_nerfs.local_planes.grad.isnan().sum() > 0:
                print("WARNING: latent_nerf grad has nan values")

            # Optimizer step
            if use_encoder and not t_args.freezes.freeze_encoder:
                optimizer_encoder.step()
                optimizer_encoder.zero_grad()

            if use_decoder and not t_args.freezes.freeze_decoder:
                optimizer_decoder.step()
                optimizer_decoder.zero_grad()

            if not t_args.freezes.freeze_lnerf:
                optimizer_latent_nerfs.step()
                optimizer_latent_nerfs.zero_grad()

            if config.global_latent_nerf.apply and not t_args.freezes.freeze_global_lnerf :
                optimizer_global_latent_nerfs.step()
                optimizer_global_latent_nerfs.zero_grad()

            if config.global_latent_nerf.apply and not t_args.freezes.freeze_base_coefs :
                optimizer_base_coefs.step()
                optimizer_base_coefs.zero_grad()
            
            pbar.update(1)

            wandb.log({
                'train_iter_time': time.time() - train_iter_time_start
            })

        #intial eval
        wandb.log({
            'video' : evaluator.get_video_eval(vae, latent_nerfs[:config.eval.video.n_scenes], latent_renderer, normalizer, batch_size=t_args.batch_size, device=device),
            'eval_metrics' : evaluator.get_evaluation_metrics(vae, latent_nerfs[:config.eval.metrics.n_scenes], latent_renderer, normalizer, batch_size=t_args.batch_size, device=device),
            'epoch': -1
        })

        cum_train_time = 0
        cum_eval_time = 0
        with tqdm.tqdm(total=t_args.n_epochs * len(dataloader)) as pbar:
            for epoch in range(t_args.n_epochs):
                pbar.set_description("/!\ DEBUGING /!\ " if debug else f"   epoch [{epoch}/{t_args.n_epochs}]")
                train_time_start = time.time()

                # Freeze tinymlp
                if epoch == t_args.freeze_tinymlp_after_n_epoch:
                    optimizer_latent_nerfs.param_groups[0]['lr'] = 0

                # Train loop
                for iter_nb, batch in enumerate(dataloader):
                    train_iter(batch)

                # Scheduler step
                if use_encoder and not t_args.freezes.freeze_encoder:
                    scheduler_encoder.step()
                if use_decoder and not t_args.freezes.freeze_decoder:
                    scheduler_decoder.step()
                if not t_args.freezes.freeze_lnerf:
                    scheduler_latent_nerfs.step()
                if config.global_latent_nerf.apply and not t_args.freezes.freeze_global_lnerf :
                    scheduler_global_latent_nerfs.step() 
                if config.global_latent_nerf.apply and not t_args.freezes.freeze_base_coefs :
                    scheduler_base_coefs.step()

                # Evaluation
                train_time_stop = time.time()
                eval_time_start = time.time()
                log_dict = {}
                if do_now(epoch, t_args.eval_video_every_epoch) : 
                    log_dict['video'] =  evaluator.get_video_eval(vae, latent_nerfs[:config.eval.video.n_scenes], latent_renderer, normalizer, batch_size=t_args.batch_size, device=device)
                    if config.global_latent_nerf.apply and (config.latent_nerf.n_local_features == 0 or config.global_latent_nerf.fusion_mode == 'sum'):
                        # In this case, the global latent nerfs can be visualized using the same tinyMLP as the scene nerfs
                        log_dict['global_nerfs_video'] =  evaluator.get_video_eval(vae, latent_nerfs.global_planes, latent_renderer, normalizer, batch_size=t_args.batch_size, device=device)
                
                if do_now(epoch, t_args.eval_metrics_every_epoch) : 
                    log_dict['eval_metrics'] = evaluator.get_evaluation_metrics(vae, latent_nerfs[:config.eval.metrics.n_scenes], latent_renderer, normalizer, batch_size=t_args.batch_size, device=device)

                if do_now(epoch, t_args.log_training_metrics_every_epoch) :
                    train_metrics = {
                        "loss": loss_meter.avg,
                        "loss_ae": loss_ae_meter.avg,
                        "loss_latent_nerf": loss_latent_nerf_meter.avg,
                        "loss_alpha_D": loss_alpha_D_meter.avg,
                        "loss_rgb_low_res": loss_rgb_low_res_meter.avg,
                        "loss_rgb_override": loss_rgb_override_meter.avg,
                        "tv" : tv_meter.avg,
                        "depth_tv" : tv_depth_meter.avg,
                        "kl_div" : kl_div_meter.avg,
                        "epoch": epoch
                    }    

                    log_dict['train_metrics'] = train_metrics

                    loss_ae_meter.reset()
                    loss_latent_nerf_meter.reset()
                    loss_alpha_D_meter.reset()
                    loss_rgb_low_res_meter.reset()
                    loss_rgb_override_meter.reset()
                    loss_meter.reset()
                    kl_div_meter.reset()
                    tv_meter.reset()
                    tv_depth_meter.reset()
                
                # Saving
                if do_now(epoch, t_args.save_every_epoch):
                    def save_model(name, save_root_path) : 
                        dict_to_save = {
                            'vae': vae.state_dict(),
                            'nerfs': latent_nerfs.state_dict(),
                            'renderer': latent_renderer.state_dict(),
                        }
                        if config.global_latent_nerf.apply:
                            dict_to_save['global_nerfs'] = latent_nerfs.global_planes.data

                        torch.save(dict_to_save, os.path.join(save_root_path, name))

                    def save_normalizer(name, save_root_path):
                        with open(os.path.join(save_root_path, name), 'wb') as outp:
                            pickle.dump(normalizer, outp, pickle.HIGHEST_PROTOCOL)
                    
                    save_root_path = os.path.join(repo_path, config.savedir, expname)
                    if not os.path.exists(save_root_path):
                        os.makedirs(save_root_path)
                    save_config(config, "config.yaml", save_root_path)
                    save_model("gvae_latest.pt", save_root_path)
                    save_normalizer("normalizer_latest.pkl", save_root_path)
                    
                    if not t_args.save_latest_only :
                        save_model(f"gvae_epoch_{epoch}.pt", save_root_path)
                        save_normalizer(f"normalizer_epoch_{epoch}.pkl", save_root_path)
                
                eval_time_stop = time.time()
                cum_train_time += train_time_stop - train_time_start
                cum_eval_time += eval_time_stop - eval_time_start

                # Logging 
                if log_dict:
                    log_dict['epoch'] = epoch
                    log_dict['cum_train_time'] = cum_train_time
                    log_dict['cum_eval_time'] = cum_eval_time
                    wandb.log(log_dict)

        wandb.finish()
        # end of train function ---------


