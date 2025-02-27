# Copyright 2024 Antoine Schnepf, Karim Kassab

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Union
import PIL.Image
import json
import os 
import math
import torch
import string
import tqdm
import torchvision.transforms as tr
import random
import numpy as np
from collections import namedtuple
from diffusers import AutoencoderKL


if "DATA_DIR" in os.environ.keys() : 
    DATA_DIR = os.environ['DATA_DIR']
else:
    DATA_DIR = "enter/path/to/data/directory"

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in [".png", ".jpeg"] # type: ignore

def get_altitude(pose) : 
    return pose[..., 11]


class Metadata:
    def __init__(
            self,
            name, 
            dataroot_dict,
            focal,
            cx, cy,
            azimuth_range,
            elevation_range,
            camera_distance,
        ) :
        self.name = name
        self.dataroot_dict = dataroot_dict
        self.focal = focal
        self.cx = cx
        self.cy = cy
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.camera_distance = camera_distance


class MultiviewDataset(torch.utils.data.Dataset):

    def __init__(
            self, 
            metadata:Metadata,
            dataroot:str,
            img_size,
            n_views:Union[int, str],
            max_obj='max',
            custom_len=None,
            shuffle:bool=False,
            shuffle_seed:int=1234,
    ) : 
        self.dataroot = dataroot
        self.metadata = metadata
        self.n_views = n_views
        self.custom_len = custom_len
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self._init_transforms(img_size)
        self._init_img_paths(dataroot, max_obj)
        self._init_poses(dataroot)
        
    def _init_transforms(self, img_size, mu=0.5, sigma=0.5): 
        self.transform = tr.Compose([
            tr.ToTensor(),
            tr.Resize(img_size, antialias=True),
            tr.CenterCrop(img_size),
            tr.Normalize([mu], [sigma])
        ])
        self.inv_transform = tr.Compose([
            tr.Normalize([-mu/sigma], [1/sigma]),
            tr.ToPILImage()
        ])

    def _init_img_paths(self, dataroot, max_obj):
        if isinstance(max_obj, str) and max_obj == 'max': 
            max_obj = np.inf

        self.obj_paths = [str(f) for f in Path(dataroot).glob('*') if not str(f).endswith(".json")]
        if self.shuffle: 
            random.Random(self.shuffle_seed).shuffle(self.obj_paths)
        self.obj_ids = [os.path.relpath(obj_path, dataroot) for obj_path in self.obj_paths]

        self.img_paths = []
        self.img_path_per_obj = {} # {obj_id : [path_1, ...] }
        self.n_img_per_obj = {}
        for i, (obj_path, obj_id) in enumerate(zip(self.obj_paths, self.obj_ids)):

            current_img_paths = [str(f) for f in sorted(Path(obj_path).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
            self.img_paths += current_img_paths
            self.img_path_per_obj[obj_id] = current_img_paths
            self.n_img_per_obj[obj_id] = len(current_img_paths)

            if i+1 >= max_obj: 
                self.obj_paths = self.obj_paths[:max_obj]
                self.obj_ids = self.obj_ids[:max_obj]
                break
        
        if self.shuffle: 
            random.Random(self.shuffle_seed).shuffle(self.img_paths)
        
        self.n_obj = len(self.obj_ids)
    
    def _init_poses(self, dataroot) : 
        poses = {}
        meta_fname = os.path.join(dataroot, 'dataset.json')
        if os.path.isfile(meta_fname):
            with open(meta_fname, 'r') as file:
                poses = json.load(file)['labels']
                if poses is not None:
                    poses = { x[0]: x[1] for x in poses }
                else:
                    poses = {}
        self.poses = poses

    def _get_pose_from_path(self, img_path) : 
        rel_path = os.path.relpath(img_path, self.dataroot)
        rel_path = rel_path.replace('\\', '/')
        return torch.tensor(self.poses.get(rel_path))

    def get_image(self, idx_obj:int, idx_img:int) : 
        obj_id = self.obj_ids[idx_obj]
        fname = self.img_path_per_obj[obj_id][idx_img]
        img = self.transform(PIL.Image.open(fname).convert('RGB'))
        pose = self._get_pose_from_path(fname)
        return dict(img=img, pose=pose)

    def get_multiview_image(self, idx_obj:int, idx_comb:int, n_views:Union[int, str], randomness:random.Random=None) :
        """Deterministic function that returns multiple views of the same object and their respective labels.
        The function uses the module random as a hash table to almost injectively map `idx_comb` to a combination
        of views, chosen among the immensly large set of possible combinations. 
        `n_view` can be either a integer or 'max', in which case the all views for the current object is returned.
        """
        obj_id = self.obj_ids[idx_obj]
        if randomness is None : 
            randomness = random.Random(hash(str(idx_comb)))

        if isinstance(n_views, str) and n_views == 'max' : 
            n_views = self.n_img_per_obj[obj_id]
            fnames = self.img_path_per_obj[obj_id]
        else : 
            fnames = randomness.sample(self.img_path_per_obj[obj_id], n_views)

        img = torch.stack([
            self.transform(PIL.Image.open(fname).convert('RGB'))
            for fname in fnames
        ])
        pose = torch.stack([
            self._get_pose_from_path(fname)
            for fname in fnames
        ])
        return dict(img=img, pose=pose)

    def __len__(self):
        "Unless `n_views` is set to 'max', the dataset has an infinite length in practice"
        if isinstance(self.n_views, str) and self.n_views == 'max' : 
            return len(self.obj_paths)
        elif self.custom_len :
            return self.custom_len
        else : 
            return sum([math.comb(n_img, self.n_views) for n_img in self.n_img_per_obj.values()]) # This can be very large and can cause an overflow error
    
    def __getitem__(self, idx) : 
        """deterministic function that returns a dict with keys 'imgs' and 'poses'
          dict['imgs']: torch.Tensor: with shape [n_views, 3, img_size, img_size] representing the images 
          dict['poses']: torch.Tensor: with shape [n_views, 25] tensor representing the camera poses
        """
        randomness = random.Random(hash(str(idx)))
        idx_obj = randomness.randint(0, self.n_obj-1)
        return self.get_multiview_image(idx_obj, idx, self.n_views, randomness)


class ObjectDataset(torch.utils.data.Dataset):
    "A dataset class for a single object / scene, providing a instance of a MultiviewDataset" 

    def __init__(self, multiview_dataset, selected_obj_idx:int=0, split='all', n_test_image=10):
        """
        args:
            selected_obj_idx: int: index of the object to be selected
            split: str: 'all' or 'train' or 'test'. 
        """

        self.multiview_dataset = multiview_dataset
        self.selected_obj_idx = selected_obj_idx
        self.inv_transform = multiview_dataset.inv_transform
        self.split = split

        self.n_test_image = n_test_image  

    def __len__(self):
        obj_id = self.multiview_dataset.obj_ids[self.selected_obj_idx]
        all_paths = self.multiview_dataset.img_path_per_obj[obj_id]

        if self.split == 'all':
            return len(all_paths)
        elif self.split == 'train':
            return len(all_paths) - self.n_test_image
        elif self.split == 'test':
            return self.n_test_image
        else:
            raise ValueError(f"Unknown split {self.split}")
    
    def _select_object(self, idx:int):
        assert idx < len(self.multiview_dataset.obj_ids)
        self.selected_obj_idx = idx

    def __getitem__(self, idx):

        if self.split == 'all':
            idx_ = idx

        elif self.split == 'train':
            if not(0 <= idx < len(self)):
                raise ValueError(f"Index {idx} is out of bounds for split '{self.split}'")
            idx_ = idx + self.n_test_image

        elif self.split == 'test':
            if not(0 <= idx < self.n_test_image):
                raise ValueError(f"Index {idx} is out of bounds for split '{self.split}'")
            idx_ = idx

        else:
            raise ValueError(f"Unknown split {self.split}")
        
        return self.multiview_dataset.get_image(self.selected_obj_idx, idx_)


class LatentDataset(torch.utils.data.Dataset):
    "A dataset class for a single object / scene that yields latent images" 
    def __init__(self, dataset:ObjectDataset, device, batch_size=64, num_workers=4, pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5', 
                 revision = 'main', probabilistic=False, normalization_args=None) : 
        self.device = device
        self.probabilistic = probabilistic
        self.source_dataset = dataset
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
        self._process_source_dataset(probabilistic, batch_size, num_workers)
        self.n_channels = self.latent_imgs.shape[-3]
        if normalization_args is None:
            normalization_args = {
                'img': {
                    'type': 'MuSigmaTanh', 
                    'tanh_scale': 0.4, 
                    'sigma_scale': 1, 
                },
                'img_mean': {
                    'type': 'MuSigmaTanh',
                    'tanh_scale': 0.6,
                    'sigma_scale': 1 ,
                },
                'img_log_std':{  
                    'type': 'MuSigmaTanh',
                    'tanh_scale': 0.6,
                    'sigma_scale': 1,
                }
            }
            print("DATASET WARNING: No normalization args provided, using default values")

        self.normalization_args = normalization_args
        self._init_transforms(probabilistic, normalization_args)

        

    @torch.no_grad()
    def _process_source_dataset(self, probabilistic, batch_size, num_workers):
        dataloader = torch.utils.data.DataLoader(self.source_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        self.vae.to(self.device)
        latent_imgs = []
        latent_img_means = []
        latent_img_log_stds = []
        for i, batch in enumerate(dataloader):
            img = batch['img'].to(self.device)
            latent_dist = self.vae.encode(img).latent_dist
            latent_imgs.append(latent_dist.sample().cpu())
            if probabilistic:
                latent_img_means.append(latent_dist.mean.cpu())
                latent_img_log_stds.append(torch.log(latent_dist.std).cpu())
            
        self.latent_imgs = torch.cat(latent_imgs, dim=0)
        if probabilistic:
            self.latent_img_means = torch.cat(latent_img_means, dim=0)
            self.latent_img_log_stds = torch.cat(latent_img_log_stds, dim=0)
        self.vae.to('cpu')

    def _get_tanh_transforms(self, alpha, eps=1e-5):
        transform = tr.Lambda(lambda x: torch.tanh(alpha*x))
        inv_transform = tr.Compose([
            tr.Lambda(lambda x : x.clip(-1 + eps, 1 - eps)),
            tr.Lambda(lambda x: 1/ alpha * torch.atanh(x))
        ])
        return transform, inv_transform

    def _get_normalize_transforms(self, mu, sigma):
        transform = tr.Normalize(mu, sigma)
        inv_transform = tr.Normalize(-mu/sigma, 1/sigma)
        return transform, inv_transform
    
    def _get_transforms(self, data, normalization_type, sigma_scale, tanh_scale):

        if normalization_type == 'MinMax':
            sigma = (data.amax(dim=[0,2,3]) - data.amin(dim=[0,2,3]))
            mu = data.amin(dim=[0,2,3])
            transform, inv_transform = self._get_normalize_transforms(mu + 0.5*sigma, sigma/2)

        elif normalization_type == 'MuSigma':
            mu = data.mean(dim=[0,2,3])
            sigma = data.std(dim=[0,2,3])
            transform, inv_transform = self._get_normalize_transforms(mu, sigma * sigma_scale)
        
        elif normalization_type == 'Tanh':
            transform, inv_transform = self._get_tanh_transforms(alpha=tanh_scale)

        elif normalization_type == 'MuSigmaTanh':
            mu = data.mean(dim=[0,2,3])
            sigma = data.std(dim=[0,2,3])
            MuSigma_tr, inv_MuSigma_tr = self._get_normalize_transforms(mu, sigma)
            Tanh_tr, inv_Tanh_tr = self._get_tanh_transforms(alpha=tanh_scale)
            transform = tr.Compose([MuSigma_tr, Tanh_tr])
            inv_transform = tr.Compose([inv_Tanh_tr, inv_MuSigma_tr])

        else:
            raise ValueError(f"Unknown normalization type {self.normalization_type}")
        
        return transform, inv_transform

    def _init_transforms(self, probabilistic, normalization_args): 
        self.rgb_inv_transform = self.source_dataset.inv_transform
        self.transform, self.inv_transform = self._get_transforms(self.latent_imgs, normalization_args['img']['type'], normalization_args['img']['sigma_scale'], normalization_args['img']['tanh_scale'])
        if probabilistic:
            self.transform_mean, self.inv_transform_mean = self._get_transforms(self.latent_img_means, normalization_args['img_mean']['type'], normalization_args['img_mean']['sigma_scale'], normalization_args['img_mean']['tanh_scale'])
            self.transform_std, self.inv_transform_std = self._get_transforms(self.latent_img_log_stds, normalization_args['img_log_std']['type'], normalization_args['img_log_std']['sigma_scale'], normalization_args['img_log_std']['tanh_scale'])

    def __len__(self):
        return len(self.source_dataset)
    
    def __getitem__(self, idx):
        res = dict(
            img = self.transform(self.latent_imgs[idx]), 
            img_rgb = self.source_dataset[idx]['img'],
            pose = self.source_dataset[idx]['pose']
        )
        if self.probabilistic:
            res['img_mean'] = self.transform_mean(self.latent_img_means[idx])
            res['img_log_std'] = self.transform_std(self.latent_img_log_stds[idx])

        return res


class MultiSceneDataset(torch.utils.data.Dataset):

    def __init__(self, datasets: list[ObjectDataset]):
        self.datasets = datasets
        self.n_scenes = len(datasets)
        self.n_img_per_scene = np.array([len(dataset) for dataset in datasets])
        self.cumlen = np.cumsum(self.n_img_per_scene)
        self.padcumlen = np.concatenate([[0], self.cumlen[:-1]])
        self.inv_transform = datasets[0].inv_transform

    def _get_scene_and_obj_idx(self, idx):
        scene_idx = np.searchsorted(self.cumlen, idx + 1)
        pad = self.padcumlen[scene_idx]
        obj_idx = idx - pad
        return scene_idx, obj_idx
    
    def __getitem__(self, idx):
        scene_idx, obj_idx = self._get_scene_and_obj_idx(idx)
        res = self.datasets[scene_idx][obj_idx]
        res.update({'scene_idx': scene_idx})
        return res

    def __len__(self):
        return self.n_img_per_scene.sum()
        

class CachedDataset(torch.utils.data.Dataset) : 

    def __init__(self, dataset, vae, device, batchsize, use_mean, repo_path, savedir) : 
        self.dataset = dataset

        rd_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self.save_dir = os.path.join(repo_path, savedir, 'buffer', f"CachedDataset_{rd_string}")

        if not os.path.exists(self.save_dir) : 
            os.makedirs(self.save_dir)

        self._process_source_dataset(vae, batchsize, num_workers=4, device=device, use_mean=use_mean)

    @torch.no_grad()
    def _process_source_dataset(self, vae, batch_size, num_workers, device, use_mean=False):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        old_device = vae.device
        vae.to(device)

        running_idx = 0
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc="Caching latents")) :
            img = batch['img'].to(device)
            latent_dist = vae.encode(img).latent_dist

            if use_mean: 
                latent_imgs = latent_dist.mean.cpu()
            else:
                latent_imgs = latent_dist.sample().cpu()

            for latent_img in latent_imgs:
                torch.save(latent_img, os.path.join(self.save_dir, f"latent_{running_idx}.pt"))
                running_idx += 1

        vae.to(old_device)
    
    def get_latents(self, idx) : 
        return torch.load(os.path.join(self.save_dir, f"latent_{idx}.pt"), map_location='cpu')

    def __len__(self) :
        return len(self.dataset)
    
    def __getitem__(self, idx) : 
        res =  self.dataset[idx]
        res.update({
            'cached_latent' : self.get_latents(idx)
        })
        return res
    

class DatasetBank:
    """Used to instantiate the dataset in a user friendly way.
    list of supported datasets:
        - shapenet_car_above
        - shapenet_car
    """


    shapenet_car_above = Metadata(
        name='shapenet_car_above',
        dataroot_dict={
            'train': os.path.join(DATA_DIR, 'cars_above_train'), 
            'test': None
        },
        focal=1.02,
        cx=0.5, cy=0.5, 
        camera_distance=1.3,
        azimuth_range=[-math.pi, math.pi],
        elevation_range=[-0.36, math.pi/2],
    )

    shapenet_car = Metadata(
        name='shapenet_car',
        dataroot_dict={
            'train' : os.path.join(DATA_DIR, 'cars_train'),
            'test' : None
        },
        focal=1.02,
        cx=0.5, cy=0.5, 
        camera_distance=1.3,
        azimuth_range=[-math.pi, math.pi],
        elevation_range=[-math.pi/2, math.pi/2],
    )


    @staticmethod
    def get_metadata(dataset_name):
        return getattr(DatasetBank, dataset_name)

    @staticmethod
    def instantiate_dataset(
            dataset_name:str,
            split:str,
            img_size:int,
            n_views:Union[int, str],
            max_obj=np.inf,
            custom_len=None,
            shuffle:bool=False,
            shuffle_seed:int=1234
        ):

        dataset_metadata = DatasetBank.get_metadata(dataset_name)
        return MultiviewDataset(
            metadata=dataset_metadata,
            dataroot=dataset_metadata.dataroot_dict[split],
            img_size=img_size,
            n_views=n_views,
            max_obj=max_obj,
            custom_len=custom_len,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed
        )
    

if __name__ == '__main__': 

    import sys
    import matplotlib.pyplot as plt
    sys.path.append("/home/a.schnepf/t23d/repo/t23d")
    os.chdir("/home/a.schnepf/t23d/repo/t23d/autoencoder")
    from ae.utils import image_grid
    dataset_name = "carla_train"
    dataset_name="shapenet_car_above"

    test_multiview_dataset = False
    if test_multiview_dataset  :
        # testing carla
        n_views=8
        dataset=DatasetBank.instantiate_dataset(
            dataset_name=dataset_name,
            split='train', 
            img_size=64,
            n_views=n_views,
            max_obj=60,
        )

        nviews_=10
        n_rows_ = 10
        rows=[]
        obj_idx = 3
        for idx in range(n_rows_//2):
        
            out = dataset.get_multiview_image(obj_idx, idx, nviews_)['img'] 

            row = image_grid([dataset.inv_transform(out[i]) for i in range(nviews_)], cols=nviews_, rows=1)
            rows.append(row)

            out = dataset.get_multiview_image(obj_idx, idx, nviews_)['img'] 

            row = image_grid([dataset.inv_transform(out[i]) for i in range(nviews_)], cols=nviews_, rows=1)
            rows.append(row)

        out1 = image_grid(rows, cols=1, rows=n_rows_)

        rows=[]
        altitudes = []
        for idx in range(10):
            out = dataset[idx]
            row = image_grid([dataset.inv_transform(out['img'][i]) for i in range(n_views)], cols=n_views, rows=1)
            rows.append(row)
        out2 = image_grid(rows, cols=1, rows=10)


        # test n_views='max'
        dataset=DatasetBank.instantiate_dataset(
            dataset_name=dataset_name,
            split='train', 
            img_size=64,
            n_views='max',
            max_obj=60,
        )

        rows=[]
        for idx in range(10):
            out = dataset[idx]
            row = image_grid([dataset.inv_transform(img) for img in out['img']], cols=len(out['img']), rows=1)
            rows.append(row)
        out3 = image_grid(rows, cols=1, rows=len(rows))

    test_latent_dataset = False
    if test_latent_dataset:
        img_size=128
        n_latent_channels=4
        mv_dataset = DatasetBank.instantiate_dataset(
            dataset_name=dataset_name,
            split='train', 
            img_size=img_size,
            n_views='max',
            max_obj=60,
        )
        obj_datasets = [
            ObjectDataset(mv_dataset, selected_obj_idx=i)
            for i in range(10)
        ]
        dataset = MultiSceneDataset(obj_datasets)

        for norm_type in ['MuSigmaTanh', 'Tanh']:
            for scale in [0.02, 0.05, 0.1, 0.2, 0.4] : 
                
            
                normalization_args = {
                        'img': {
                            'type': norm_type, 
                            'tanh_scale': scale, 
                            'sigma_scale': 1, 
                        },
                        'img_mean': {
                            'type': 'MuSigmaTanh',
                            'tanh_scale': 0.6,
                            'sigma_scale': 1 ,
                        },
                        'img_log_std':{  
                            'type': 'MuSigmaTanh',
                            'tanh_scale': 0.6,
                            'sigma_scale': 1,
                        }
                    }
                
                latent_dataset = LatentDataset(dataset, device='cuda:0', batch_size=64, num_workers=4, probabilistic=True, normalization_args=normalization_args)
                
                # plotting histograms of the normalized data points

                def stack_channels(dataset, key='img', transform=None, mask=None): 
                    colors_per_channel = {
                        i : []
                        for i in range(n_latent_channels)
                    }
                    if mask is None:
                        mask = torch.ones_like(dataset[0][key][0]).bool()

                    for k in range(len(dataset)):
                        #imgarr = dataset._inv_transform(dataset[k][key]).numpy()

                        if not transform:
                            imgarr = dataset[k][key].numpy()
                        else:
                            imgarr = transform(dataset[k][key]).numpy()

                        for i in range(n_latent_channels):
                            colors_per_channel[i].append(imgarr[i][mask].flatten())

                    for i in range(n_latent_channels):
                        colors_per_channel[i] = np.concatenate(colors_per_channel[i], axis=0)
                    return colors_per_channel

                def _plot_hist(ax, key, inv_transform, mask=None):
                    data_norm = stack_channels(latent_dataset, key=key, mask=mask)
                    data_raw = stack_channels(latent_dataset, key=key, transform=inv_transform, mask=mask)
                    colors=['red', 'green', 'blue', 'orange']
                    normalized_mean = []
                    for k in range(4):
                        ax[0].hist(data_norm[k], bins=50, density=True, label=f"channel {k}", color=colors[k], alpha=0.3)
                        ax[0].set_title(f"{key} after normalization of type {latent_dataset.normalization_args[key]['type']}")
                        normalized_mean.append(data_norm[k].mean())
                        ax[1].hist(data_raw[k], bins=50, density=True, label=f"channel {k}", color=colors[k], alpha=0.3)
                        ax[1].set_title(f"{key} without normalization ")
                    return normalized_mean
                bg_mask = np.zeros(shape=(img_size//8, img_size//8)).astype(bool)
                pad = 2
                bg_mask[:pad, :] = 1
                bg_mask[-pad:, :] = 1
                bg_mask[:, :pad] = 1
                bg_mask[:, -pad:] = 1
                plt.figure()
                plt.imshow(bg_mask)
                plt.axis('off')
                plt.title('mask for the background')
                plt.show()
                def to_pixel_values(L) : 
                    return [(x+1)/2 for x in L]
                for mask in [None, bg_mask] : 
                    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
                    img_log_std_per_channel_mean = _plot_hist(axes[0], 'img_log_std', latent_dataset.inv_transform_std, mask=mask)
                    img_mean_per_channel_mean = _plot_hist(axes[1], 'img_mean', latent_dataset.inv_transform_mean, mask=mask)
                    img_per_channel_mean = _plot_hist(axes[2], 'img', latent_dataset.inv_transform, mask=mask)
                print(f"normalization type: {norm_type}   tanh_scale: {scale}")
                print(f"current mean bg color (in [0,1]): {to_pixel_values(img_per_channel_mean)}")
                print("-----------------")
                #plt.legend()
                #plt.show()


    
    plot_std_img_grid = False
    if plot_std_img_grid:
        inv_tr = tr.ToPILImage()
        n_imgs = 20
        imgs = []
        for idx in range(n_imgs):
            std_pil_img = inv_tr(latent_dataset[idx]['img_log_std'][:3]*2 -1 ).resize((100,100))
            mu_pil_img = inv_tr(latent_dataset[idx]['img_mean'][:3]*2 -1 ).resize((100,100))
            rgb_pil_img = latent_dataset.rgb_inv_transform(latent_dataset[idx]['img_rgb'][:3]).resize((100,100))
            imgs.append(image_grid([std_pil_img, mu_pil_img, rgb_pil_img], 1, 3))
        imgs = image_grid(imgs, 4, n_imgs//4)
        imgs