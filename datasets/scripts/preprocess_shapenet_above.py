# script to create a new version of shapenet where all cars are seen from above by removing all views from below

import os
import json 
import tqdm 

ALTITUDE_THRESHOLD = -0.5

def get_pose_from_path(self, img_path, dataroot) : 
    rel_path = os.path.relpath(img_path, dataroot)
    rel_path = rel_path.replace('\\', '/')
    return self.poses.get(rel_path)

def get_altitude(pose):
    return pose[11]

dataroot = os.path.join(os.environ['DATA_DIR'], 'cars_train')
new_dataroot = os.path.join(os.environ['DATA_DIR'], 'cars_above_train')

meta_fname = os.path.join(dataroot, 'dataset.json')
assert os.path.isfile(meta_fname)
with open(meta_fname, 'r') as file:
    poses = json.load(file)['labels']
    poses = { x[0]: x[1] for x in poses }


obj_ids = [p for p in os.listdir(dataroot) if not p.endswith(".json")]
for obj_id in tqdm.tqdm(obj_ids): 

    local_dirs = [
        'rgb',
        'pose',
        'intrinsics',
    ]
    to_copy = "intrinsics.txt"

    for to_copy_dir in local_dirs : 
        os.makedirs(os.path.join(new_dataroot, obj_id, to_copy_dir), exist_ok=True)
    os.system(f"cp {os.path.join(dataroot, obj_id, to_copy)} {os.path.join(new_dataroot, obj_id)}")
    
    for img_name in os.listdir(os.path.join(dataroot, obj_id, 'rgb')):

        pose = poses[os.path.join(obj_id, 'rgb', img_name)]
        if get_altitude(pose) > ALTITUDE_THRESHOLD : 
            img_id = img_name.split(".png")[0]
            os.system(f"cp {os.path.join(dataroot, obj_id, 'rgb', img_name)} {os.path.join(new_dataroot, obj_id, 'rgb')}")
            os.system(f"cp {os.path.join(dataroot, obj_id, 'pose', img_id+'.txt')} {os.path.join(new_dataroot, obj_id, 'pose')}")
            os.system(f"cp {os.path.join(dataroot, obj_id, 'intrinsics', img_id+'.txt')} {os.path.join(new_dataroot, obj_id, 'intrinsics')}")