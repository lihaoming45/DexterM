# TPNP: Generating Dexterous Grasping Motion with Contact Latent Diffusion and Temporal Parametric Neural Pose Optimization

## Project Page
https://lihaoming45.github.io/tpnp/index.html

## Dataset Download
To download the dataset, please send us an e-mail (haomingli@zju.edu.cn) including contact details (title, full name, organization, and country) and the purpose for downloading the dataset. Important note for students and post-docs: we hope to know the contact details of your academic supervisor. By sending the e-mail you accept the following terms and conditions.

### Terms and Conditions
When downloading and utilizing the TPNP dataset, you are required to carefully review and adhere to the following terms and conditions. By proceeding with the download and usage, you acknowledge that you have read, understood, and agreed to these terms. Any breach of this agreement will result in the immediate termination of your rights under this license. The dataset is developed by the State Key Laboratory of Industrial Control Technology at Zhejiang University, which retains all copyright and patent rights.

#### Terms of Use
- The dataset is strictly limited to non-commercial academic research and educational purposes.
- Any other applications, including but not limited to integration into commercial products, use in commercial services, or further development of commercial projects, are strictly prohibited.
- Modification, resale, or redistribution of the dataset is not allowed without prior written consent.
- Proper citation of the associated paper is required when the dataset or its concepts are used.

# Dexterous Motion Dataset
-  The dataset is available for research purposes at  [ [TPNP-DexGrasp](xxx) ]
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
# Visualization
Below you can see some generated results from the proposed TPNP:
![Grasp Motion](images/TPNPDataset_github.gif)

# Citation
```
@article{wang2022dexgraspnet,
  title={DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation},
  author={Wang, Ruicheng and Zhang, Jialiang and Chen, Jiayi and Xu, Yinzhen and Li, Puhao and Liu, Tengyu and Wang, He},
  journal={arXiv preprint arXiv:2210.02697},
  year={2022}
}
```
