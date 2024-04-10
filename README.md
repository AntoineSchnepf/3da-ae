todo before code release:
- check licences
- discuss the editing of the envrionment.yaml file // generate a new one using a cleaner install of depandencies


# Exploring 3D-aware Latent Spaces for Efficiently Learning Numerous Scenes
**Official paper implementation accepted to CVPR 2024 3DMV Workshop**
> Antoine Schnepf*, Karim Kassab*, Jean-Yves Franceschi, Laurent Caraffa, Flavian Vasile, Jeremie Mary, Andrew Comport, Valérie Gouet-Brunet (* indicates equal contribution)<br>
| [Webpage](https://3da-ae.github.io/) | [Full Paper](https://arxiv.org/abs/2403.11678) |<br>

Abstract: *We present a method enabling the scaling of NeRFs to learn a large number of semantically-similar scenes. We combine two techniques to improve the required training time and memory cost per scene. First, we learn a 3D-aware latent space in which we train Tri-Planes scene representations, hence reducing the resolution at which scenes are learned. Moreover, we present a way to share common information across scene representations, hence allowing for a reduction of model complexity to learn a particular scene. Our method reduces effective per-scene memory costs by 44% and per-scene time costs by 86% when training 1000 scenes.*


![LatentScenes](https://github.com/AntoineSchnepf/3da-ae/assets/85931369/50862207-2868-4718-955b-7c473cf12f72)



## Setup
In this section we detail how to prepare the dataset and the environment for training and exploiting 3Da-AE.

### Environment 
Our code has been tested on:
- Linux (Debian)
- Python 3.11.5
- Pytorch 2.0.1
- CUDA 11.8
- `L4` and `A100` NVIDIA GPUs


You can use Anaconda to create the environment:
```
conda create --name 3daae -y python=3.11.5
conda activate 3daae
```
Then, you can install pytorch with Cuda 11.8 using the following command:
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 --upgrade
```
_You may have to adapt the cuda version according to your hardware, we recommend using CUDA >= 11.8_

To install the remaining requirements, execute:
```
pip install -r requirements.txt
```

### Dataset

First, set where to save the dataset by exporting the DATA_DIR variable as an environment variable:

```
export DATA_DIR=path/for/data/directory
```

Then, download the renderings of the car category of [ShapeNet](https://shapenet.org/) (data is hosted by the authors of [Scene Representation Networks](https://www.vincentsitzmann.com/srns/)):


```
cd datasets/scripts
python run_me.py 
```


## Usage
You can now train your 3D-aware autoencoder on the shapenet car dataset:
```
python train.py --config train.yaml
```

Then, you can learn new scenes in the latent space of 3Da-AE using:
```
python exploit.py --config exploit.yaml
```
## Visualization / evaluation
We visualize and evaluate our method using [wandb](https://wandb.ai/site). You can get quickstarted [here](https://docs.wandb.ai/quickstart).

## Citation

If you find this research project useful, please consider citing our work:
```
@article{schnepf2024exploring,
      title={Exploring 3D-aware Latent Spaces for Efficiently Learning Numerous Scenes}, 
      author={Antoine Schnepf and Karim Kassab and Jean-Yves Franceschi and Laurent Caraffa and Flavian Vasile and Jeremie Mary and Andrew Comport and Valérie Gouet-Brunet},
      journal={arXiv preprint arXiv:2403.11678},
      year={2024}
}
```

