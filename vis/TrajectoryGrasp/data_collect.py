import numpy as np
from glob import glob
import os.path as osp
def dexgraspnet_data_collect(path):
    obj_files_path = glob(osp.join(path,'*.npy'))
    object_code_list = [osp.basename(file).split('.')[0] for file in obj_files_path]
    return object_code_list

def obman_data_collect(path):
    obj_files_path = glob(osp.join(path,'*'))
    object_code_list = [osp.basename(file).split('.')[0] for file in obj_files_path]



