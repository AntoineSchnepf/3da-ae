# Exploring 3D-aware Latent Spaces for Efficiently Learning Numerous Scenes
**Official paper implementation submitted to CVPR 2024 3DMV Workshop**
> Antoine Schnepf*, Karim Kassab*, Jean-Yves Franceschi, Laurent Caraffa, Flavian Vasile, Jeremie Mary, Andrew Comport, Valérie Gouet-Brunet (* indicates equal contribution)<br>
| [Webpage](https://3da-ae.github.io/) | [Full Paper](https://arxiv.org/abs/2403.11678) |<br>

Abstract: *We present a method enabling the scaling of NeRFs to learn a large number of semantically-similar scenes. We combine two techniques to improve the required training time and memory cost per scene. First, we learn a 3D-aware latent space in which we train Tri-Planes scene representations, hence reducing the resolution at which scenes are learned. Moreover, we present a way to share common information across scene representations, hence allowing for a reduction of model complexity to learn a particular scene. Our method reduces effective per-scene memory costs by 44% and per-scene time costs by 86% when training 1000 scenes.*


![LatentScenes](https://github.com/AntoineSchnepf/3da-ae/assets/85931369/50862207-2868-4718-955b-7c473cf12f72)


![Teaser image](assets/schema.svg)




## Setup
In this section we detail how to prepare the dataset and the environment for training and exploiting 3Da-AE.

## Requirements

## Usage


##

## Citation

```
@article{schnepf2024exploring,
      title={Exploring 3D-aware Latent Spaces for Efficiently Learning Numerous Scenes}, 
      author={Antoine Schnepf and Karim Kassab and Jean-Yves Franceschi and Laurent Caraffa and Flavian Vasile and Jeremie Mary and Andrew Comport and Valérie Gouet-Brunet},
      journal={arXiv preprint arXiv:2403.11678},
      year={2024}
}
```

