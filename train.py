import torch
import os
from prodict import Prodict
import argparse
import sys
import pickle

from ae.utils import load_config
from ae.window_normalizer import WindowNormalizer
from ae.trainers import train, init_models, init_datasets
from ae.eval import Evaluator
# ----------------------------------------------------------------------

if __name__ == '__main__' :
    REPO_PATH=os.path.dirname(os.path.realpath(__file__))

    # --- args parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="train_default.yaml")
    parser.add_argument('--config_dir', type=str, default=os.path.join(REPO_PATH, "configs"))
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    config = Prodict.from_dict(load_config(args.config, args.config_dir, from_default=True, default_cfg_name="train_default.yaml"))
    assert config.pretrain_exp_name != config.exp_name, "pretrain_exp_name and exp_name must be different"

    # --- initialiasing data ---
    train_scenes, multi_scene_trainset, test_scenes, multi_scene_testset, pose_sampler = init_datasets(config)

    # --- initialiasing models ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vae, latent_renderer, latent_nerfs = init_models(config, device)
    criterion = torch.nn.MSELoss()
    evaluator = Evaluator(train_scenes, test_scenes, pose_sampler, config, REPO_PATH)
    normalizer = WindowNormalizer(config.vae.normalization.img)
    print(f"--using torch.device: {device}--")

    latent_renderer.to(device)
    vae.to(device)

    # Pretraining script for latent nerf
    if config.pretrain:
        os.system(f"rm -f -r {os.path.join(REPO_PATH, config.savedir, 'buffer', '*')}")
        train(
            config=config, 
            t_args=config.pretrain_args,
            expname=config.pretrain_exp_name, 
            multi_scene_set=multi_scene_trainset, 
            vae=vae, 
            latent_renderer=latent_renderer, 
            latent_nerfs=latent_nerfs, 
            normalizer=normalizer, 
            criterion=criterion, 
            evaluator=evaluator, 
            device=device, 
            repo_path=REPO_PATH,
            debug=False
        )

    # load the latent nerfs, the latent renderer and the normalizer
    load_path = os.path.join(REPO_PATH, config.savedir, config.pretrain_exp_name)
    checkpoint = torch.load(
        os.path.join(load_path, "gvae_latest.pt"),
        map_location=torch.device('cpu')
    )
    vae.load_state_dict(checkpoint['vae'])
    latent_renderer.load_state_dict(checkpoint['renderer'])
    latent_nerfs.load_state_dict(checkpoint['nerfs'])

    with open(os.path.join(load_path, "normalizer_latest.pkl"), 'rb') as inp:
        normalizer = pickle.load(inp)


    # training loop
    if config.train:
        os.system(f"rm -r {os.path.join(REPO_PATH, config.savedir, 'buffer', '*')}")
        train(
            config=config, 
            t_args=config.train_args,
            expname=config.exp_name, 
            multi_scene_set=multi_scene_trainset, 
            vae=vae, 
            latent_renderer=latent_renderer, 
            latent_nerfs=latent_nerfs, 
            normalizer=normalizer, 
            criterion=criterion, 
            evaluator=evaluator, 
            device=device, 
            repo_path=REPO_PATH,
            debug=False
        )



