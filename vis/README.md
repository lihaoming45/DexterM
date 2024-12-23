## Demo
We provide two different ways for visualization:
- For the .html file , running the following command: 
  ```Shell
  python vis/vis_html.py
  ```
- For the isaacgym simulator,  running the following command: 
  ```Shell
  python vis/vis_isaac.py --task=IsaacGraspSimulator --seed=0 --rl_device=cpu --sim_device=cpu --seq_id=[0,1,2] --object_code=core-cellphone-ff9e8c139527c8b85f16c469ffeb982e --test --headless
  ```
