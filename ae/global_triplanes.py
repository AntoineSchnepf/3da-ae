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

'''
    Class to implement Tri-Planes from local/global representations.
    Each plane has n_local_features feature planes specific to it,
    and is concatinated with n_global_features feature planes.
    The global feature planes are computed from a weighted sum of n_base global planes.
'''
import torch

class TriplaneManager(torch.nn.Module) : 

    def __init__(
            self, 
            n_scenes,
            local_nerf_cfg,
            global_nerf_cfg,
            device_='cpu'
        ) : 
        super(TriplaneManager, self).__init__()
        
        # init
        self.global_enabled = global_nerf_cfg.apply
        self.n_scenes = n_scenes
        self.n_base = global_nerf_cfg.n_base
        self.n_global_features = global_nerf_cfg.n_global_features
        self.n_local_features = local_nerf_cfg.n_local_features
        if self.global_enabled :
            self.n_local_features = global_nerf_cfg.n_local_features
        self.triplane_resolution = local_nerf_cfg.triplane_resolution
        self.fusion_mode = global_nerf_cfg.fusion_mode
        self.bias = global_nerf_cfg.bias
        self.device_=device_

        self.local_planes = torch.nn.Parameter(
            torch.randn(size=(self.n_scenes, 3, self.n_local_features, self.triplane_resolution, self.triplane_resolution)),
            requires_grad=True
        )

        if self.global_enabled: 
            self.global_planes = torch.nn.Parameter(
                torch.randn(size=(self.n_base, 3, self.n_global_features, self.triplane_resolution, self.triplane_resolution)),
                requires_grad=True,
            )

            self.coefs = torch.nn.Parameter(
                torch.randn(size=(self.n_scenes, self.n_base)), 
                requires_grad=True
            )
            
            if self.bias: 
                self.bias = torch.nn.Parameter(
                    torch.zeros(size=(self.n_scenes, 1, 1, 1, 1)), 
                    requires_grad=True
                )
            else : 
                self.bias = 0

        # sanity check
        if self.fusion_mode == 'sum' :
            assert self.local_planes.shape[-3] == self.global_planes.shape[-3], "Local and global planes must have the same number of features when using sum fusion mode"

    def fuse(self, p1, p2) : 
        if self.fusion_mode == 'concat' : 
            return torch.cat((p1, p2), -3)
        elif self.fusion_mode == 'sum' :   
            return p1 + p2
        else : 
            raise NotImplementedError
        
    def compute_orthogonal_regularization(self) : #TODO: add in cfg / train
        dot_products =  torch.einsum(
            "nijkl,mijkl->nm",
            [self.global_planes, self.global_planes]
        )
        return torch.sum(torch.abs(dot_products))   
    
    def get_triplanes(self, scene_idxs) :
        "returns the triplanes for the given list of scenes_idxs"

        local_ = self.local_planes[scene_idxs].to(self.device_)

        if self.global_enabled :
            coefs_ = self.coefs[scene_idxs].to(self.device_)
            global_ = torch.einsum(
                "ib,b...->i...",
                [coefs_, self.global_planes.to(self.device_)]
            )

            bias_ = self.bias[scene_idxs].to(self.device_)
            return self.fuse(local_, global_ + bias_)
        
        return local_
    
    def __getitem__(self, scene_idxs) : 

        if isinstance(scene_idxs, int) : 
            return self.get_triplanes([scene_idxs]).squeeze(0)
        
        return self.get_triplanes(scene_idxs)


