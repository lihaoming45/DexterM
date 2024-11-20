# Asset Process

This folder is for processing object models. For ShadowHand, the objects are based on the DexGraspNet. For the MANO hand, the objects are based on Obman, GRAB, and ContactPose.
For each object we filer out non-manifolds and models of small volumes, and calculate the **sign ditacne field (sdf)** as the input of out TPNP optimization.

## Download Object Meshes
  - #### Download object models from DexGraspNet
    - -  Downloading the sources of object models for the following websites
    - -  [ShapeNetCore](https://shapenet.org/)
    - -  [ShapeNetSem](https://shapenet.org/)
    - -  [Mujoco](https://github.com/kevinzakka/mujoco_scanned_objects)
    - -  [DDG](https://gamma.umd.edu/researchdirections/grasping/differentiable_grasp_planner)(Deep Differentiable Grasp)

    - Following the guidance of the [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet/tree/main/asset_process) Extraction to download the object mesh.
      ```bash
      # ShapeNetCore
      python extract.py --src data/ShapeNetCore.v2 --dst data/raw_models --set core # replace data root with yours
      # ShapeNetSem
      python extract.py --src data/ShapeNetSem/models --dst data/raw_models --set sem --meta data/ShapeNetSem/metadata.csv
      # Mujoco
      python extract.py --src data/mujoco_scanned_objects/models --dst data/raw_models --set mujoco
      # DDG
      python extract.py --src data/Grasp_Dataset/good_shapes --dst data/raw_models --set ddg
      ```
      
   - #### Download object models from Obman
      - Following the guidance of the [Obman](https://hassony2.github.io/obman) website to download the object mesh.

   - #### Download object models from ContactPose
      - Following the guidance of the [ContactPose](https://github.com/facebookresearch/ContactPose/blob/main/docs/doc.md#downloading-data) to download the object mesh.
        
        ```bash
        $ python scripts/download_data.py --type 3Dmodels
        $ python scripts/download_data.py --type markers
        ```

   - #### Download object models from GRAB
     -  Following the guidance of [GRAB website](https://grab.is.tue.mpg.de/) to download the object models.
     -  Using the following command from [GRAB repository](https://github.com/otaheri/GRAB)  to extract the ZIP files
        ```bash
        python grab/unzip_grab.py   --grab-path $PATH_TO_FOLDER_WITH_ZIP_FILES \
                                    --extract-path $PATH_TO_EXTRACT_GRAB_DATASET_TO
        ```


## Object Models Preprocessing
- The following packages need to be installed
    -  trimesh
    -  [mesh_to_sdf](https://github.com/marian42/mesh_to_sdf)
    -  mcubes
    -  hydra
    -  numpy
    -  [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus)
    -  [CoACD](https://github.com/SarahWeiii/CoACD)

- After downloading the object mesh, we recommend you to follow the pipeline of [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet/tree/main/asset_process) to process these models, including 
  -  Extraction: Organize models into a folder.
  -  Manifold: Use ManifoldPlus to convert raw models into manifolds robustly.
  -  Normalization: Adjust centers and sizes of models. Then filter out bad models.
  -  Decomposition: Use CoACD to decompose models and export urdf files for later physical simulation.

- Then, running the following command to calculate the object sdf.  
    ```bash
    python ./mesh_to_sdf.py --obj-dataset-path $DATASET_PATH \
                            --dataset-name  $DATASET_NAME 
    ```

- The structure of the final object dataset for shadow hand is:
```bash
ShadowHand
+-- source(-category)-code0
|  +-- coacd
|  |  +-- coacd.urdf
|  |  +-- decomposed.obj
|  |  +-- decomposed.sdf
|  |  ...
+-- source(-category)-code1
...
```
- The structure of the final object dataset for human hand is:
```bash
HumanHand
+-- Obman
|  +-- source(-category)-code0
|  |  +-- source(-category)-code0-scale.obj
|  |  +-- source(-category)-code0-scale.sdf
|  ...
+-- source(-category)-code1
...
+-- ContactPose
|  +-- object_name
|  |  +-- object_name.obj
|  |  +-- object_name.sdf
|  ...
...
+-- GRAB
|  +-- object_name
|  |  +-- object_name.obj
|  |  +-- object_name.sdf
|  ...
...


