import os,gc
import cv2
from glob import glob
from utils.torch_jit_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
import open3d as o3d
import sys
import os.path as osp
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from DexRep_Isaac.dexgrasp.utils.rot6d import get_trans_oth6d_according_to_rotmat, joint_names, dexrep_hand,\
    fingertips,normalize_vector, cross_product
from torchvision.io import write_video
class IsaacGraspSimulator:

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.table_height = cfg['env']['table_height']
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent

        self.up_axis = 'z'
        self.fingertips = fingertips
        self.hand_center = ["robot0:palm"]
        self.num_fingertips = len(self.fingertips)
        # self.object_scale = 1
        self.joint_names =joint_names
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.device = "cpu"
        self.device_type = device_type
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device_id = cfg.get("device_id", 0)
            self.device = "cuda" + ":" + str(self.device_id)

        self.dexrep_hand = dexrep_hand
        self.gym_initialization(cfg)

    def gym_initialization(self, cfg):

        self.gym = gymapi.acquire_gym()
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.headless = cfg["headless"]
        self.graphics_device_id = cfg.get("graphics_device_id", self.device_id)
        print("self.device_id:",self.device_id)
        print("self.graphics_device_id",self.graphics_device_id)
        if  self.headless == True:
            self.graphics_device_id = -1

        self.num_states = cfg["env"].get("numStates", 0)
        self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

    def clean_sim(self):
        if self.headless == False:
            self.gym.destroy_viewer(self.my_viewer)
        self.gym.destroy_sim(self.sim)
        del self.grasp_data_dct
        del self.root_state_tensor
        del self.dof_state
        gc.collect()

        a=1

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.video_frames_list = []
        self.camera_rgb_tensor_list = []
        self.my_viewer=None
        self.enable_viewer_sync = True
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        self.gym.prepare_sim(self.sim)

        if self.headless == False:
            self.create_viewer()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.gym_refresh_state()


    def input_load(self, cfg, data_list=None):
        self.cfg = cfg
        self.table_height = cfg['env']['table_height']
        self.init_hand_pos_z = cfg['env']['init_hand_pos_z']
        self.load_grasp_data(cfg, data_list)
        self.load_object()
        self.cfg["env"]["numEnvs"] = len(self.grasp_data_dct['grasp_seqs'])

    def create_viewer(self):
        # subscribe to keyboard shortcuts
        self.my_viewer= self.gym.create_viewer(self.sim,
                        gymapi.CameraProperties())

        self.gym.subscribe_viewer_keyboard_event(
            self.my_viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.my_viewer, gymapi.KEY_V, "toggle_viewer_sync")
        # set the camera position based on up axis
        sim_params = self.gym.get_sim_params(self.sim)
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            # cam_pos = gymapi.Vec3(10, 10, 3.0)
            # cam_target = gymapi.Vec3(0, 0, 0.0)
            cam_pos = gymapi.Vec3(-0.6, -0.6, 1.2)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.6)
        else:
            cam_pos = gymapi.Vec3(10, 10, 3.0)
            cam_target = gymapi.Vec3(0, 0, 0.0)

        self.gym.viewer_camera_look_at(self.my_viewer, None, cam_pos, cam_target)

        self.seq_id_str = 'all' if self.cfg['seq_id'] is  None else str(self.cfg['seq_id'])
        self.viewer_video_path = osp.join(self.cfg['env']['video_save_root'],self.cfg['object_code'])
        self.img_save_path = osp.join(self.cfg['env']['img_save_root'],self.cfg['object_code'])
        os.makedirs(self.img_save_path,exist_ok=True)
        os.makedirs(self.viewer_video_path,exist_ok=True)

    def load_camera(self,env_ptr, cam_pos=None,cam_target=None):
        cam_pos = gymapi.Vec3(-0.6, -0.6, 1.1)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.6)
        self.cam_width = 1024
        self.cam_height = 768
        self.img_buf = torch.zeros(
            (self.num_envs,self.cam_height, self.cam_width, 3), device=self.device, dtype=torch.float)
        self.camera_properties = gymapi.CameraProperties()
        self.camera_properties.horizontal_fov = 90.0  # 调整视野角度
        self.camera_properties.width = self.cam_width
        self.camera_properties.height = self.cam_height
        self.camera_properties.enable_tensors = True

        cam_handle = self.gym.create_camera_sensor(env_ptr, self.camera_properties)
        self.gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)
        raw_rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)

        rgb_tensor = gymtorch.wrap_tensor(raw_rgb_tensor)
        rgb_tensors=[]
        rgb_tensors.append(rgb_tensor)
        self.camera_rgb_tensor_list.append(rgb_tensors)

    def render(self, sync_frame_time=False):
        if self.my_viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.my_viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.my_viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.my_viewer, self.sim, True)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

            else:
                self.gym.poll_viewer_events(self.my_viewer)
        else:
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def load_grasp_data(self,cfg):
        data_dct_list = np.load(osp.join(cfg['env']['grasp_root'], f"{cfg['object_code']}.npy"), allow_pickle=True).item()
        grasp_seqs = data_dct_list['grasp_seqs']
        obj_scales = data_dct_list['obj_scale']
        obj_rotmat = data_dct_list['obj_rotmat']

        if cfg['seq_id'] is not None:
            grasp_seqs = grasp_seqs[cfg['seq_id']]
            obj_scales = obj_scales[cfg['seq_id']]
            obj_rotmat = obj_rotmat[cfg['seq_id']]

        self.grasp_data_dct = {"grasp_seqs":torch.tensor(grasp_seqs),'obj_scale': obj_scales, 'obj_rotmat':obj_rotmat}
        self.num_envs = len(self.grasp_data_dct['grasp_seqs'])


    def load_object(self):
        object_code = self.cfg['object_code']
        self.object_code_list = [self.cfg['object_code']]
        mesh_root = "/home/mani/Downloads/meshdata/{}/coacd/".format(object_code)
        obj_mesh = trimesh.load_mesh(osp.join(mesh_root, "decomposed.obj"))

        self.obj_half_height_list = []
        for i in range(self.num_envs):
            scale = self.grasp_data_dct['obj_scale'][i]
            obj_rotmat = self.grasp_data_dct['obj_rotmat'][i]
            obj_half_height = self.get_obj_half_height(obj_mesh, scale, obj_rotmat)
            self.obj_half_height_list.append(obj_half_height)
        a=1

    def _load_object_asset(self, assets_path):
        object_asset_dict = {}
        self.num_object_bodies_list = []
        self.num_object_shapes_list = []
        # mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]
        # self.obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj"]
        self.obj_asset_root =self.cfg["env"]["obj_asset_root"]
        self.raw_obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"]
        self.grasp_root = self.asset_root + '/grasp_gt/'

        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 1  # 500
            object_asset_options.fix_base_link =False #False
            # object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            # object_asset_options.override_com = True
            # object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            object_asset_file = "coacd.urdf"
            object_asset = self.gym.load_asset(self.sim, osp.join(self.obj_asset_root,object_code,'coacd'), object_asset_file, object_asset_options)
            object_body_name = self.gym.get_asset_rigid_body_names(object_asset)[0]
            a=1
            # self.grasp_gt_data = np.load(os.path.join(self.grasp_root, f"{object_code}.npy"), allow_pickle=True)
            # self.object_scale = self.grasp_gt_data[70]['scale']

            if object_asset is None:
                print(object_code)
            assert object_asset is not None

            self.num_object_bodies_list.append(self.gym.get_asset_rigid_body_count(object_asset))
            self.num_object_shapes_list.append(self.gym.get_asset_rigid_shape_count(object_asset))
            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)
            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
            object_asset_dict[object_id] = object_asset
            object_asset_dict['object_code'] = object_code
        object_asset_dict['object_body_name'] = object_body_name
        return object_asset_dict

    def _load_shadow_hand_asset(self):
        asset_root = "../../assets"
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        table_texture_files = "/home/mani/unstacking_model_verify/dex_visualization/DexRep_Isaac/assets/textures/texture_wood_brown_1033760.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)
        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True #True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        # asset_options.override_com = True
        # asset_options.override_inertia = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 1 #10
        asset_options.linear_damping = 1 #10
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        # tendon set up
        limit_stiffness =1 #1 #50  # 30
        t_damping =10  #10 #0.05  # 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in
                              range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in
                                     actuated_dof_names]
        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)
        return shadow_hand_asset, shadow_hand_dof_props, table_texture_handle

    def _load_table_asset(self):
        table_dims = gymapi.Vec3(1, 1, self.table_height)
        # self.table_height = table_dims.z
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())
        return table_asset, table_dims

    def get_obj_half_height(self,obj_mesh,object_scale, object_rotmat):
        obj_mesh.vertices = np.matmul(obj_mesh.vertices, object_rotmat.T)
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(obj_mesh.faces)
        pcd = mesh_o3d.sample_points_poisson_disk(1024)
        pcd = np.asarray(pcd.points)

        # rot_pcd = np.matmul(self.obj_pcd, object_rotmat.T)
        min_z = np.min(pcd[:, 2])
        obj_half_height = min_z * object_scale
        return obj_half_height

    def set_obj_start_root_state(self, obj_half_height,object_rotmat):
        r = R.from_matrix(object_rotmat)
        rot_quat = r.as_quat()

        object_z = self.table_height - obj_half_height
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, object_z)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])
        return object_start_pose

    def set_hand_start_root_state(self,grasp_data_start_pose=None):
        if grasp_data_start_pose is not None:
            start_position = grasp_data_start_pose[:3]
            start_euler = grasp_data_start_pose[3:6]
            r = R.from_euler('XYZ', [start_euler[0], start_euler[1], start_euler[2]], degrees=False)
            rot_quat = r.as_quat()
        else:
            # self.init_hand_pos_z = 1.0
            start_position = np.array([0, 0, self.init_hand_pos_z])
            rot_quat = np.array([0.,0.,0.,1.])
        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*start_position)  # gymapi.Vec3(0.1, 0.1, 0.65)
        shadow_hand_start_pose.r = gymapi.Quat(rot_quat[0],rot_quat[1], rot_quat[2],rot_quat[3])
        return shadow_hand_start_pose

    def set_table_start_root_state(self,table_dims):
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        self.table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

    def set_hand_color(self, env_ptr,shadow_hand_actor):
        hand_color = [147 / 255, 215 / 255, 160 / 255]
        hand_rigid_body_index = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19, 20],
                                 [21, 22, 23, 24, 25]]
        for n in self.agent_index[0]:
            for m in n:
                for o in hand_rigid_body_index[m]:
                    self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL, gymapi.Vec3(*hand_color))

    def set_object_table_color(self, env_ptr, object_handle, table_handle):
        object_color = [90 / 255, 94 / 255, 173 / 255]
        self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
        table_color = [100 / 254, 120 / 254, 167 / 254]
        self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))

    def set_friction(self,env_ptr, table_handle, object_handle):
        table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
        object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
        table_shape_props[0].friction = 1.0
        for i in range(len(object_shape_props)):
            object_shape_props[i].friction =15 #15.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
        self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

    def add_env_instance(self, i, kwargs):
        shadow_hand_asset, shadow_hand_start_pose = kwargs["shadow_hand_asset"],kwargs["shadow_hand_start_pose"]
        shadow_hand_dof_props = kwargs['shadow_hand_dof_props']
        object_asset_dict, object_start_pose = kwargs["object_asset_dict"],kwargs["object_start_pose"]
        table_asset = kwargs["table_asset"]

        object_idx_this_env = i % len(self.object_code_list)
        # create env instance
        env_ptr = self.gym.create_env(self.sim, self.lower, self.upper, self.num_per_row)
        max_agg_bodies = self.num_shadow_hand_bodies + self.num_object_bodies_list[object_idx_this_env] + 2
        max_agg_shapes = self.num_shadow_hand_shapes + self.num_object_shapes_list[object_idx_this_env] + 2

        if self.aggregate_mode >= 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

        # load shadow hand  for each env
        shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose,
                                                  "hand", i, -1, 0)

        self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
        hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
        self.hand_indices.append(hand_idx)

        # randomize colors and textures for rigid body
        self.set_hand_color(env_ptr, shadow_hand_actor)

        # load object for each env
        object_handle = self.gym.create_actor(env_ptr, object_asset_dict[object_idx_this_env], object_start_pose,
                                              "object", i, 0, 0)
        object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
        self.object_indices.append(object_idx)

        self.gym.set_actor_scale(env_ptr, object_handle, kwargs['object_scale'])  ####

        # add table
        table_handle = self.gym.create_actor(env_ptr, table_asset, self.table_pose, "table", i, -1, 0)
        # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, kwargs['table_texture_handle'])
        table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
        self.table_indices.append(table_idx)

        # set friction
        self.set_friction(env_ptr, table_handle, object_handle)

        # set object and table color
        self.set_object_table_color(env_ptr, object_handle,table_handle)

        if self.aggregate_mode > 0:
            self.gym.end_aggregate(env_ptr)

        self.envs.append(env_ptr)
        self.shadow_hands.append(shadow_hand_actor)

        # load camera
        if self.cfg["headless"]==False:
            self.load_camera(env_ptr)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self.num_per_row = num_per_row


        self.lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        self.upper = gymapi.Vec3(spacing, spacing, spacing)

        # create hand asset
        shadow_hand_asset, shadow_hand_dof_props, table_texture_handle = self._load_shadow_hand_asset()
        # create object asset
        object_asset_dict = self._load_object_asset('../assets')

        self.object_body_name = self.gym.get_asset_rigid_body_names(object_asset_dict[0])[0]
        self.object_body_idx = self.gym.find_asset_rigid_body_index(object_asset_dict[0], self.object_body_name)

        # create table asset
        table_asset, table_dims = self._load_table_asset()

        # set the original root of table
        self.set_table_start_root_state(table_dims)


        self.shadow_hands, self.hand_indices,  self.envs = [], [], []
        self.object_indices,  self.table_indices = [], []
        for i in range(self.num_envs):
            object_scale = self.grasp_data_dct['obj_scale'][i] #
            object_rotmat = self.grasp_data_dct['obj_rotmat'][i]

            obj_half_height = self.obj_half_height_list[i]

            # add object half_height to the grasp data
            if self.cfg['env']['grasp_data_process']:
                self.grasp_data_dct['grasp_seqs'][i] = self.add_object_half_height(self.grasp_data_dct['grasp_seqs'][i],obj_half_height)
            # set the original root axis of hand
            shadow_hand_start_pose = self.set_hand_start_root_state()

            # set the original root of object
            object_start_pose = self.set_obj_start_root_state(obj_half_height, object_rotmat)

            env_input = {"shadow_hand_asset":shadow_hand_asset,"shadow_hand_start_pose":shadow_hand_start_pose,
                 "object_start_pose":object_start_pose,'object_asset_dict':object_asset_dict,'object_scale':object_scale,
                 "table_asset":table_asset, "table_texture_handle":table_texture_handle,
                 'shadow_hand_dof_props':shadow_hand_dof_props}
            self.add_env_instance(i, env_input)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

    def gym_refresh_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def pre_physics_step(self, actions, id):

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.actions = actions.clone().to(self.device)
        self.cur_targets[:] = self.actions
        self.cur_targets[:] = self.act_moving_average * self.cur_targets[:] + (1.0 - self.act_moving_average) * self.prev_targets[:]
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        if id == -1:
            a=1
        else:
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.cur_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
            a=1

    def camera_image_save(self, frame_idx):
        # 转换成图片格式
        os.makedirs(self.img_save_path, exist_ok=True)
        img = self.img_buf[0].cpu().numpy()
        cv2.imwrite(f"{self.img_save_path}/frame_{frame_idx:04d}.png", img)

    def viewer_image_save(self,frame_idx):
        self.gym.write_viewer_image_to_file(self.my_viewer, f"{self.viewer_video_path}/{self.seq_id_str}_{frame_idx:03d}.png")
        a=1

    def single_env_video_write(self):
        for i in range(self.num_envs):
            seq_id = self.cfg['seq_id'][i]
            frame_list_i = [img[i]for img in self.video_frames_list]
            video_frame_i = torch.stack(frame_list_i,dim=0) #N_f,H,W,C
            write_video(f"{self.img_save_path}/seq_id{seq_id}_video.mp4", video_frame_i, fps=20)

    def viewer_video_write(self):
        frames_paths = sorted(glob(osp.join(self.viewer_video_path, "*.png")),key=lambda name: int(name[-7:].split('.')[0]))
        frame_list = [cv2.imread(img_path) for img_path in frames_paths]
        frames_tensor = torch.tensor(np.stack(frame_list,axis=0))
        write_video(f"{self.viewer_video_path}/{self.seq_id_str}_video.mp4", frames_tensor, fps=20)
        [os.remove(img_path) for img_path in frames_paths]

        a=1

    def compute_pixel_obs(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=self.device).view(3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=self.device).view(3, 1, 1)
        self.im_size = self.camera_properties.width
        for i in range(self.num_envs):
            self.img_buf[i] = self.camera_rgb_tensor_list[i][0][:,:,:3].float()
        self.video_frames_list.append(self.img_buf.clone())

        # view_img_buf= self.viewer_img
        # self.viewer_camera_rgb_img_list.append(view_img_buf)
        self.gym.end_access_image_tensors(self.sim)
        # pixel_obs = torch.flatten(self.img_buf, start_dim=1, end_dim=-1)
        # return pixel_obs.cpu().numpy()

    def step(self, actions, id):
        self.id = id
        self.pre_physics_step(actions, id)
        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

    def add_lift_action(self,grasp_data):
        """
        :param grasp_data: (B,N,31)
        :return:
        """
        B,N,D = grasp_data.shape
        # --------------- define clench increment ----------------
        increment = np.zeros_like(grasp_data[:,39,:], dtype=grasp_data.dtype) #(B,1,31)
        increment[...,9:] = (grasp_data[:,39, 9:] - grasp_data[:, 38, 9:]) * 0.5 ##(B,22)
        # static_data =  np.ones((B, 10, 31), dtype=grasp_data.dtype)*grasp_data[:,:1,:]

        extended_data = np.ones((B, 40, 31), dtype=grasp_data.dtype)*grasp_data[:,-1:,:]

        # --------------- add clench action ----------------
        for i in range(5):
            extended_data[:,i] = extended_data[:,i - 1] + increment*0.1

        # --------------- add lift trajectory ----------------
        for i in range(5, 40):
            extended_data[:,i] = extended_data[:,i - 1]
            extended_data[:,i, 2] += 0.01
        grasp_data = np.concatenate([grasp_data, extended_data],axis=1)
        return grasp_data

    def add_table_height_to_data(self,grasp_data):
        grasp_data[..., 2] -=(self.init_hand_pos_z-self.table_height)

        return grasp_data


    def add_object_half_height(self, grasp_data, object_half_height):
        # object_half_height = torch.tensor(self.obj_half_height_list).view(self.num_envs,1)
        grasp_data[..., 2] -=object_half_height
        return grasp_data

    def robust_compute_euler_from_ortho6d(self, pose):
        """
        :param pose: (B,6)
        :return eulers: (B,3)
        """
        hand_rotations = self.robust_compute_rotation_matrix_from_ortho6d(torch.tensor(pose))
        eulers = [R.from_matrix(rot).as_euler('XYZ', degrees=False) for rot in hand_rotations]
        eulers = np.stack(eulers,axis=0) #(B,3)
        return torch.tensor(eulers).float()

    def robust_compute_quat_from_ortho6d(self, pose):
        """
        :param pose: (B,6)
        :return quats: (B,3)
        """
        hand_rotations = self.robust_compute_rotation_matrix_from_ortho6d(torch.tensor(pose))
        quats = [R.from_matrix(rot).as_quat() for rot in hand_rotations]
        quats = np.stack(quats,axis=0) #(B,4)
        return torch.tensor(quats).float()

    def robust_compute_rotation_matrix_from_ortho6d(self, poses):
        """
        Instead of making 2nd vector orthogonal to first
        create a base that takes into account the two predicted
        directions equally
        """
        if len(poses.size())==1:
            poses = poses.unsqueeze(0)

        x_raw = poses[:, 0:3]  # batch*3
        y_raw = poses[:, 3:6]  # batch*3

        x = normalize_vector(x_raw)  # batch*3
        y = normalize_vector(y_raw)  # batch*3
        middle = normalize_vector(x + y)
        orthmid = normalize_vector(x - y)
        x = normalize_vector(middle + orthmid)
        y = normalize_vector(middle - orthmid)
        # Their scalar product should be small !
        # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
        z = normalize_vector(cross_product(x, y))

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        matrix = torch.cat((x, y, z), 2)  # batch*3*3
        # Check for reflection in matrix ! If found, flip last vector TODO
        return matrix

    def sequence_rotation(self, grasp_data, aug_rotmat):
        """
        :param grasp_data: (B,N,31)
        :return:
        """
        B, N, D = grasp_data.shape
        batch_trans_oth6d = grasp_data.reshape(-1,D)[:,:9]
        new_trans_oth6d = get_trans_oth6d_according_to_rotmat(batch_trans_oth6d, aug_rotmat) #(B*N,9)
        grasp_data[:,:,:9] = new_trans_oth6d.view(B,N,9).numpy()
        return grasp_data

    def grasping_filter_index(self, grasp_data):
        B, N, D = grasp_data.shape
        grasp_init_pos_z = grasp_data[:,0,2] #(B,)

        select_idxs = grasp_init_pos_z>0.02
        # grasp_data = grasp_data[select_idxs]
        return select_idxs

    def joint_vel_control(self, grasp_data, control_weight=0.7):
        #grasp_data (B,N,31)
        control_id = -2
        joint_start = grasp_data[:,0:1,9:]
        joint_motion_gap = grasp_data[:,:control_id,9:]-joint_start#(B,N-2,22)

        joint_weighted_motion = joint_motion_gap*control_weight +joint_start #(B,N-2,22)
        grasp_data[:, :control_id, 9:] = joint_weighted_motion
        return grasp_data


    def grasp_data_process(self, grasp_data, cfg):

        if cfg['env']['joint_vel_control']:
            grasp_data = self.joint_vel_control(grasp_data)

        # ------ rotation augmentation on grasp sequence ------
        grasp_data = self.sequence_rotation(grasp_data,cfg['aug_rotmat'])

        # ------ filter out those  grasp under the object ------
        select_idxs = self.grasping_filter_index(grasp_data)

        # ------add the half object height to the grasp sequence ------
        # grasp_data = self.add_object_half_height(grasp_data)

        # ------add the table height to the grasp sequence ------
        grasp_data = self.add_table_height_to_data(grasp_data)
        # ------add the lift object sequence ------
        grasp_data = self.add_lift_action(grasp_data) #(B,N,31)

        B, N, D = grasp_data.shape
        batch_oth6d = grasp_data.reshape(-1,D)
        batch_eulers = self.robust_compute_euler_from_ortho6d(batch_oth6d[:,3:9]).view(B, N, 3)
        new_grasp_data =torch.zeros(B, N, 3+3+22)# torch.zeros(B, N, 3+3+18)

        new_grasp_data[...,:3] = torch.tensor(grasp_data[...,:3])
        new_grasp_data[...,3:6] = batch_eulers

        angles = grasp_data[..., 9:].astype(np.float32) #(B,N,, 22)

        new_grasp_data[...,6:] = torch.tensor(angles)

        return new_grasp_data , select_idxs #(B, N ,28), (B,)

    def data_select(self,sim_select_idxs):
        self.grasp_data_dct['grasp_seqs'] = self.grasp_data_dct['grasp_seqs'][sim_select_idxs.cpu().numpy()]
        self.grasp_data_dct['obj_scale'] = self.grasp_data_dct['obj_scale'][sim_select_idxs.cpu().numpy()]
        self.grasp_data_dct['obj_rotmat'] = self.grasp_data_dct['obj_rotmat'][sim_select_idxs.cpu().numpy()]

    def grasp_data_save(self, sim_select_idxs):
        self.data_select(sim_select_idxs)
        # self.grasp_data_dct['grasp_seqs'] = self.grasp_data_dct['grasp_seqs'][sim_select_idxs]
        # self.grasp_data_dct['obj_scale'] = self.grasp_data_dct['obj_scale'][sim_select_idxs]
        # self.grasp_data_dct['obj_rotmat'] = self.cfg['aug_rotmat']

        save_folder ='rid{}'.format(self.cfg['aug_id'])
        os.makedirs(osp.join(self.cfg['env']['grasp_save_root'],save_folder),exist_ok=True)

        save_file_name = osp.join(self.cfg['object_code']+'-rid{}.npy'.format(self.cfg['aug_id']))

        save_path = osp.join(self.cfg['env']['grasp_save_root'],save_folder, save_file_name)
        np.save(save_path, np.asarray(self.grasp_data_dct))
        a=1

    def run(self):
        grasp_data = self.grasp_data_dct['grasp_seqs'] #(B,N,28)
        grasp_data = grasp_data.transpose(0,1)
        obj_actor_pos_z_start_state = self.root_state_tensor.view(self.num_envs, -1, 13)[:, 1, :3].clone()  # (B,3)
        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))

        for i, element in enumerate(grasp_data):
            with torch.no_grad():
                # actions = self.vec_env.ori() #(B, 24)
                actions = torch.zeros(self.num_envs,28)
                actions[:,:] = torch.tensor(element,device=actions.device)
                # -------- refresh the realtime state of the simulator -------------
                if i == 0:
                    self.cur_targets[:] = actions
                    self.dof_state[:, 0] = self.cur_targets.view(-1)
                    self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                                          gymtorch.unwrap_tensor(all_hand_indices),
                                                          len(all_hand_indices))
                    a=1
                else:
                    self.step(actions, id=0)
                self.gym_refresh_state()

                if self.cfg["headless"]==False:
                    if self.cfg['env']['img_save'] or self.cfg['env']['viewer_video_save']:
                        self.compute_pixel_obs()
                    if i>1 and self.cfg['env']['img_save']:
                        self.camera_image_save(i)
                    if i>1 and self.cfg['env']['viewer_video_save']:
                        self.viewer_image_save(i)
                    a=1


                obj_actor_pos_z_state = self.root_state_tensor.view(self.num_envs,-1,13)[:,1,:3] #(B,3)
        lift_height = obj_actor_pos_z_state[:,-1]- obj_actor_pos_z_start_state[:,-1] #（B，）

        sim_select_idxs = (lift_height>0.10)
        return sim_select_idxs

    def vis_run(self,cfg):
        self.cfg = cfg
        self.input_load(cfg)
        self.create_sim()
        sim_select_idxs = self.run()
        self.save_select_idxs(sim_select_idxs)
        if self.cfg['env']['viewer_video_save']:
            self.single_env_video_write()
            self.viewer_video_write()
            a=1

    def save_select_idxs(self,sim_select_idxs):
        save_path = osp.join(self.cfg['env']['grasp_root'],'success_idxs')
        os.makedirs(save_path,exist_ok=True)
        save_file = osp.join(save_path,'{}_idx.npy'.format(self.cfg['object_code']))

        np.save(save_file,np.asarray(sim_select_idxs))



