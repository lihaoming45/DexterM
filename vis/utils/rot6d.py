import torch

joint_names = ['robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0', 'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1',
               'robot0:MFJ0', 'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0', 'robot0:LFJ4', 'robot0:LFJ3',
               'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0', 'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1',
               'robot0:THJ0']
dexrep_hand = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal",
               "robot0:ffmiddle", "robot0:mfmiddle", "robot0:rfmiddle", "robot0:lfmiddle", "robot0:thmiddle",
               "robot0:ffproximal", "robot0:mfproximal", "robot0:rfproximal", "robot0:lfmetacarpal",
               "robot0:thproximal"]
fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal","robot0:thdistal"]

def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
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
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out

def robust_compute_ortho6d_from_rotmat(rotation):
    return rotation.T.ravel()[:6]



def get_trans_oth6d_according_to_rotmat(trans_oth6d, rotmat):
    """
    :param trans_oth6d: #(N, 9)
    :param rotmat: (3,3)

    :return: new_trans_oth6d: #(N, 9)
    """
    N = len(trans_oth6d)
    trans_oth6d = trans_oth6d if torch.is_tensor(trans_oth6d) else torch.tensor(trans_oth6d)
    trans, oth6d = trans_oth6d[:, :3].float(), trans_oth6d[:, 3:9].float()

    rotmat = rotmat if torch.is_tensor(rotmat) else torch.tensor(rotmat)
    new_trans = torch.mm(trans, rotmat.T.float())  # (N,3)

    rot = robust_compute_rotation_matrix_from_ortho6d(oth6d)  # (N,3,3)
    new_target_rot = torch.bmm(rotmat.float().unsqueeze(0).repeat(N,1,1), rot)  # (N,3,3)
    new_oth6d = torch.stack([rot_i[:, :2].T.ravel()[:6] for rot_i in new_target_rot], dim=0)  # (N,6)

    new_trans_oth6d = torch.cat([new_trans, new_oth6d], dim=1)  # (N,9)
    return new_trans_oth6d

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out
