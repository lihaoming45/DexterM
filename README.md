# GraspM<sup>3</sup>: Dexterous Grasp `M`otion Generation at `M`illion Scale with Se`m`antic Labelling
The dataset has been released, including the grasping motion sequences of Shadow robot hands and mano hand for more than 8,000 objects. 
In **Version 1**, we randomly generated 150–200 grasping trajectories per object in arbitrary directions, ensuring collision-free interactions while maintaining natural and smooth hand postures. For each object, approximately 150–200 grasping trajectories were ultimately generated. Subsequently, we leveraged the Isaac Gym simulation environment to augment the data and filter out successful grasping trajectories, leading to the release of **Version 2**. Finally, using a large language model (LLM), we generated semantic annotations for different grasping trajectories based on multi-view rendered images of hand-object interactions, culminating in the release of **Version 3**.

- **Visualization of multi-object grasping**
  <img src="images/viewer_video_num441_final3_1.gif" width=120% />


- **Visualization of different grasping trajectories for the same object**
     | Cube | Camera|Light-Bulb|
     | :---: | :---: |:---: |
     | <img src="images/viewer_Rcube.gif" width="95%"> | <img src="images/viewer_camera.gif" width="95%"> | <img src="images/viewer_lightbulb.gif" width="95%"> |


**More visualization can be found in this [page](https://lihaoming45.github.io/GraspM3/index.html)!**

# Download
To download the dataset, please send us an e-mail (haomingli@zju.edu.cn) including contact details (title, full name, organization, and country) and the purpose for downloading the dataset. Important note for students and post-docs: We hope to know your academic supervisor's contact details. By sending the e-mail you accept the following terms and conditions. 

For object models, we offer two access options：
- Asset processing for **object models**. See folder [asset_process](./asset_process).
-  directly downloading from **"ObjectMeshes"** folder in the dataset link, but please cite the relevant references ([Citation](#Citation)).


## Terms and Conditions
When downloading and utilizing the TPNP dataset, you are required to carefully review and adhere to the following terms and conditions. By proceeding with the download and usage, you acknowledge that you have read, understood, and agreed to these terms. Any breach of this agreement will result in the immediate termination of your rights under this license. The dataset is developed by the State Key Laboratory of Industrial Control Technology at Zhejiang University, which retains all copyright and patent rights.

### Terms of Use
- The dataset is strictly limited to non-commercial academic research and educational purposes.
- Any other applications, including but not limited to integration into commercial products, use in commercial services, or further development of commercial projects, are strictly prohibited.
- Modification, resale, or redistribution of the dataset is not allowed without prior written consent.
- Proper citation of the associated paper is required when the dataset or its concepts are used.

# Dataset Description

-  The dataset for dexterous multifigured robotic hands containing more than 8,000 objects and 1,364,360 trajectories, has three different versions.
   - **Version 1 (v1)**  contains optimized grasp trajectory sequences generated using the TPNP method, including:
      - 1,152,000 trajectories based on the ShadowHand.
      -  212,360 trajectories based on the MANO hand.
      -  8152 object models.

   - **Version 2 (v2)**  enhances and filters the trajectory data from Version 1 using [IsaacGym](https://github.com/isaac-sim/IsaacGymEnvs). Specifically, data augmentation is performed in the simulation environment by applying rotational transformations around the object's center. Then, the grasp sequences from Version 1 are simulated and filtered using the Isaac simulation environment, eliminating samples where the grasp attempts failed.The default setting can be found in[vis/isaac_test](./vis/isaac_test). For each crawl sequence, we've added the following information:
       - the annotation of whether it can be successfully executed in simulation.
       - the rotation matrix for the object model.
   - **Version 3 (v3)**  uses large language models (LLMs) to provining semantic annotation of trajectory samples inclduing
      - A long text description.
      - Functional attributes.
      - Object categories.
      - Grasp directions
      - Contact areas.
   

## Datas File Structure
- Our working file structure is as:
```bash
GraspM<sup>3</sup> Dataset
ObjectMeshes
+-- meshdata_mano
+-- meshdata_shadow

+-- ShadowHand
|  +-- v1 # The first version of our Released dataset.
|  |  +--source(-category)-code0.npy The filename denotes the object ID of the ShapeNet.
|  |  +--source(-category)-code1.npy
|  |  +-- ...
|  +-- v2 # The second version of our Released dataset.
|  +-- v3 # The third version of our Released dataset.
+-- HumanHand
|  +-- v1 # The first version of our Released dataset.
|  |  +--source(-category)-code0-scale The filename denotes the object ID of the Obman and The decimal at the end of the file name indicates the scale of the object.
|  |  +--source(-category)-code1-scale
|  |  |  +--p0.npy each .npy file store a grasp pose represented by MANO parameters
|  |  |  +--p1.npy
|  |  |  +--p2.npy
|  |  |  +-- ...
|  |  +-- ...

|  +-- v2 # The second version of our Released dataset.
|  +-- v3 # The third version of our Released dataset.
```
# Quick Visualization Example
- #### Requirements
  This visualization process has the following requirements:
  - Numpy
  - Python >=3.6.0
  - torch>=1.10.1
  - trimesh
  - [secenepic](https://microsoft.github.io/scenepic/python/index.html)
  - [manotorch](https://github.com/lixiny/manotorch.git)
    
- #### Visualizing the hand and object meshes for each frame

    To visualize and grasp trajectory, run the *vis/visualize_html.py*
    
    ```Shell
    python vis/visualize_html.py --data-path $TPNP_DATASET_PATH \
                                      --model-path $MANO_MODEL_FOLDER
    ```

Below you can see some generated results from the GraspM<sup>3</sup>:
![Grasp Motion](images/TPNPDataset_github.gif)

# Citation
   This work can be considered a significant extension of our conference paper, Contact2Grasp: 3D grasp synthesis via hand-object contact constraint, published in the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI '23)
```
@article{li2023contact2grasp,
  title={Contact2grasp: 3d grasp synthesis via hand-object contact constraint},
  author={Li, Haoming and Lin, Xinzhuo and Zhou, Yang and Li, Xiang and Huo, Yuchi and Chen, Jiming and Ye, Qi},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={1053--1061},
  year={2023}
}
```
We also kindly ask you to cite Wang et al. [DexGraspNet website](https://pku-epic.github.io/DexGraspNet/), Taheri et al. [GRAB website](https://grab.is.tue.mpg.de/), Hasson et al. [Obman website](https://www.di.ens.fr/willow/research/obman/data/) and Brahmbhatt et al.[ContactPose website](https://github.com/facebookresearch/ContactPose) whose object meshes are used for our TPNP dataset. 
