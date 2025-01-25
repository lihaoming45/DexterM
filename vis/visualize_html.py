import torch
import numpy as np
import trimesh, os
from psbody.mesh import Mesh as PMesh
from psbody.mesh.colors import name_to_rgb
from shadowpth.utils.hand_model import HandModel
from shadowpth.utils.object_model import ObjectModel
from shadowpth.utils.vis_utils import sp_animation


def makepath(desired_path, isfile=True):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


class html_visualizer:
    def __init__(self, dataset_path=./DexGraspNet/v1/'):
        self.device = 'cpu'
        self.dataset_path = dataset_path
        self.object_model = ObjectModel(
            data_root_path='./meshdata',
            batch_size_each=1,
            num_samples=2000,
            device=self.device
        )

        self.hand_model = HandModel(
            mjcf_path='./shadowpth/mjcf/shadow_hand_wrist_free.xml',
            mesh_path=./shadowpth/mjcf/meshes',
            contact_points_path='./shadowpth/mjcf/contact_points.json',
            penetration_points_path='./shadowpth/mjcf/penetration_points.json',
            n_surface_points=2000,
            device=self.device
        )
        self.tra_save_path = './vis/html_save/'

    def save(self, sam_ps, idx, object_code, obj_mesh):

        sam_ps = torch.tensor(sam_ps,device=self.device,dtype=torch.float32)
        self.hand_model.set_parameters(sam_ps)
        grasp_anim = sp_animation()
        for j in range(sam_ps.shape[0]):
            shadow_meshes_i = self.hand_model.get_trimesh_data(j)
            shadow_meshes_i = trimesh.util.concatenate(shadow_meshes_i)
            shandow_pmesh = PMesh(v=shadow_meshes_i.vertices, f=shadow_meshes_i.faces, vc=name_to_rgb['pink'])
            obj_pmesh = PMesh(v=obj_mesh.vertices, f=obj_mesh.faces, vc=np.array([0.0, 1.0, 1.0]))
            grasp_anim.add_frame([shandow_pmesh, obj_pmesh], ['frame{}'.format(str(j)), 'obj'])

        save_path = makepath(os.path.join(self.tra_save_path, "{}_id{}.html".format(object_code, str(idx))))
        grasp_anim.save_animation(save_path)

    def data_load(self, object_code):
        seq_data = np.load(os.path.join(self.dataset_path, object_code + '.npy'), allow_pickle=True).tolist()
        self.object_model.initialize(object_code)
        return seq_data

    def get_object_mesh(self, object_scale):
        self.object_model.object_scale_tensor = torch.tensor(object_scale, dtype=torch.float,
                                                             device=self.device).reshape(1, 1)
        obj_mesh = self.object_model.get_trimesh_data(0)
        obj_mesh.visual.vertex_colors = [0, 255, 255, 255]

        return obj_mesh

    def run(self, object_code):
        seq_data = self.data_load(object_code)

        for idx, seq_data_i in enumerate(seq_data):
            seq_params, obj_scale = seq_data_i['seq_qpos'], seq_data_i['scale']
            obj_mesh = self.get_object_mesh(obj_scale)
            self.save(seq_params, idx, object_code, obj_mesh)

        a = 1


if __name__ == '__main__':
    visalizer = html_visualizer()
    visalizer.run('core-bottle-3b26c9021d9e31a7ad8912880b776dcf')

