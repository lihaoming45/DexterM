"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize hand model using plotly.graph_objects
"""

import os
import sys

os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath('.'))

import numpy as np
import torch
import trimesh as tm
import transforms3d
import plotly.graph_objects as go
from shadowpth.utils.hand_model import HandModel

from grasping_lhm.utils.visual import pcd_instance, mesh_instance
import open3d.visualization as o3dv
from open3d import geometry as o3dg
from open3d import utility as o3du
from shadowpth.utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if __name__ == '__main__':
    device = torch.device('cpu')

    # hand model

    hand_model = HandModel(
        mjcf_path='./mjcf/shadow_hand_wrist_free_win.xml',
        mesh_path='./mjcf/meshes',
        contact_points_path='./mjcf/contact_points.json',
        penetration_points_path='./mjcf/penetration_points.json',
        n_surface_points=2000,
        device=device
    )
    # joint_angles = torch.tensor([0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0], dtype=torch.float, device=device)
    joint_angles = torch.zeros(22, dtype=torch.float, device=device)
    # joint_angles[7]+=0.2
    # rotation0 = torch.tensor(transforms3d.euler.euler2mat(0.0, -np.pi / 103, 0.7, axes='rxyz'), dtype=torch.float, device=device)
    rotation0 = torch.tensor(transforms3d.euler.euler2mat(0.0, 0, 0., axes='rxyz'), dtype=torch.float, device=device)
    rotation1 = torch.tensor(transforms3d.euler.euler2mat(-np.pi / 2, 0.0, 0.0, axes='rxyz'), dtype=torch.float, device=device)


    hand_pose0 = torch.cat([torch.tensor([0, 0, 0], dtype=torch.float, device=device), rotation0.T.ravel()[:6], joint_angles])
    hand_model.set_parameters(hand_pose0.unsqueeze(0))
    hand_mesh = hand_model.get_trimesh_data(0)
    shadow_meshes0 = tm.util.concatenate(hand_mesh)
    shadow_mesh0_open3d = mesh_instance(shadow_meshes0.vertices,shadow_meshes0.faces)
    # info
    surface_points = hand_model.get_surface_points()[0].detach().cpu().numpy()
    contact_candidates = hand_model.get_contact_candidates()[0].detach().cpu().numpy()
    penetration_keypoints = hand_model.get_penetraion_keypoints().detach().cpu()
    penetration_keypoints = torch.bmm(torch.tensor(rotation1).unsqueeze(0), penetration_keypoints.transpose(1,2))
    penetration_keypoints = penetration_keypoints.transpose(1, 2)

    shadow_pcd = pcd_instance(penetration_keypoints)

    palm_meshes = hand_model.get_trimesh_data(0)[0]
    # palm_meshes.export('./shadow_palm.obj')
    palm_verts = palm_meshes.vertices
    vert_colors = np.ones_like(palm_verts)*np.array([[0.8, 0.1, 0]])

    vert_colors[:20,:] = np.array([[0.1,0.8,0.]])

    FOR1 = o3dg.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    palm_mesh = mesh_instance(palm_meshes.vertices,palm_meshes.faces)
    face_normal =np.asarray(palm_mesh.triangle_normals)
    palm_normal = face_normal[98]

    palm_normal =torch.mm(torch.tensor(rotation1,dtype=torch.float32), torch.tensor(palm_normal,dtype=torch.float32).unsqueeze(1))
    palm_normal = palm_normal.transpose(1,0).squeeze().numpy()

    palm_center = palm_meshes.vertices[palm_meshes.faces[98]].mean(axis=0)

    points = np.asarray([palm_center, palm_center+palm_normal])
    line = [[0,1]]
    norm_line = o3dg.LineSet()
    norm_line.lines = o3du.Vector2iVector(line)
    norm_line.points = o3du.Vector3dVector(points)


    # joints_pcd = pcd_instance(palm_verts)
    # joints_pcd.colors = o3du.Vector3dVector(vert_colors)

    # o3dv.draw_geometries([FOR1, norm_line])#norm_line
    o3dv.draw_geometries([FOR1])#norm_line

    
    print('n_surface_points', surface_points.shape[0])
    print('n_contact_candidates', contact_candidates.shape[0])

    # visualize

    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', with_contact_points=False)
    hand_trimesh = hand_model.get_trimesh_data(i=0)

    surface_points_plotly = [go.Scatter3d(x=surface_points[:, 0], y=surface_points[:, 1], z=surface_points[:, 2], mode='markers', marker=dict(color='lightblue', size=2))]
    contact_candidates_plotly = [go.Scatter3d(x=contact_candidates[:, 0], y=contact_candidates[:, 1], z=contact_candidates[:, 2], mode='markers', marker=dict(color='white', size=2))]
    penetration_keypoints_plotly = [go.Scatter3d(x=penetration_keypoints[:, 0], y=penetration_keypoints[:, 1], z=penetration_keypoints[:, 2], mode='markers', marker=dict(color='red', size=3))]
    for penetration_keypoint in penetration_keypoints:
        mesh = tm.primitives.Capsule(radius=0.01, height=0)
        v = mesh.vertices + penetration_keypoint
        f = mesh.faces
        penetration_keypoints_plotly += [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='burlywood', opacity=0.5)]

    fig = go.Figure(hand_plotly + surface_points_plotly + contact_candidates_plotly + penetration_keypoints_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.write_html('./shaodowhand_flat.html')
    # fig.show()
    a=1
