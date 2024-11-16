# TPNP
Generating Dexterous Grasping Motion with Contact Latent Diffusion and Temporal Parametric Neural Pose Optimization

Below you can see some generated results from the proposed TPNP:
|![Grasp Motion](images/TPNPDataset_github.gif)
# Dexterous Motion Dataset
- The grasp motion dataset of the shadow hand is based on the object from the [ [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet.git) ]
-  The dataset is available for research purposes at  [ [TPNP-DexGrasp](https://onedrive.live.com/?id=991F1C6479E03435%21s1a592872606f490c926ed916a44dabb7&cid=991F1C6479E03435) ]
- Our working file structure is as:

```bash
TPNPDataset
+-- HumanHand
|  +-- v1 # The first version of our Released dataset.
|  |  +--04074963_f677657b2cd55f930d48ff455dd1223_0.2375 The filename denotes the object ID of the Obman and The decimal at the end of the file name indicates the scale of the object.
|  |  +--04074963_f677657b2cd55f930d48ff455dd1223_0.125
|  |  |  +--p0.npy each .npy file store a grasp pose represented by MANO parameters
|  |  |  +--p1.npy
|  |  |  +--p2.npy
|  |  |  +-- ...
|  |  +-- ...

|  +-- v2 # The second version of our Released dataset.

+-- ShadowHand
|  +-- v1 # The first version of our Released dataset.
|  |  +--core-bottle-1cf98e5b6fff5471c8724d5673a063a6.npy The filename denotes the object ID of the ShapeNet.
|  |  +--core-jar-85b34acd44a557ae21072d05c97a5e0a.npy
|  |  +-- ...
|  +-- v2 # The second version of our Released dataset.

```


# Citation
```
@article{wang2022dexgraspnet,
  title={DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation},
  author={Wang, Ruicheng and Zhang, Jialiang and Chen, Jiayi and Xu, Yinzhen and Li, Puhao and Liu, Tengyu and Wang, He},
  journal={arXiv preprint arXiv:2210.02697},
  year={2022}
}
```
