# This file should test byol encoder by showing the nearest neighbor
# Load the data and split it into train and test set
import os
import re
import glob
import torch
import pickle
from pathlib import Path
from examples.dataset import RetargetingTrajectoryDataset, get_retargeting_train_val
from third_person_man.utils import VISION_IMAGE_MEANS, VISION_IMAGE_STDS, crop_transform, crop_demo, convert_to_demo_frame_ids


def get_demo_length(data_directory, view_num):
    demo_length = [] 
    roots = sorted(glob.glob(f'{data_directory}/demonstration_*'))
    demo_dict = crop_demo(data_directory, view_num)
    demo_frame_ids = convert_to_demo_frame_ids(demo_dict)
    for demo_index, demo in enumerate(roots):
        pattern = r'\d+$'
        demo_num = int(re.findall(pattern, demo)[0])
        clipping_index = demo_frame_ids[str(demo_num)]
        demo_action_ids = _get_demo_action_ids(data_directory, clipping_index, demo_num, view_num)
        single_demo_length = demo_action_ids[1] - demo_action_ids[0]
        demo_length.append(single_demo_length)
    return demo_length

def _get_demo_action_ids(data_directory, image_frame_ids, demo_num, view_num):
    action_ids = []
    # Will traverse through image indices and return the idx that have the image_frame_ids
    image_indices_path = os.path.join(data_directory,'demonstration_{}'.format(demo_num),'image_indices_cam_{}.pkl'.format(view_num))
    with open(image_indices_path, 'rb') as file:
        image_indices = pickle.load(file)

    i = 0
    for action_id, (demo_id, image_id) in enumerate(image_indices):
        if image_id == image_frame_ids[i]:
            action_ids.append(action_id)
            i += 1
            
        if i == 2: 
            break

    return action_ids



def main():
    # initialize calibrate_fingertips
    data_directory = Path('/data/irmak/third_person_manipulation/mustard_picking')
    demo_length = get_demo_length(data_directory, 1)
    print(demo_length)
    train_data, test_data = get_retargeting_train_val(data_directory, vqbet_get_future_action_chunk = False,  view_num = 1)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, pin_memory=False
    )
    for idx, data in enumerate(train_loader):
        # obs, act, goal = (x.to(cfg.device) for x in data)
        if idx >= 10: break
        obs, act = (x.to('cuda') for x in data)
        print(act)
    


if __name__ == '__main__':
    main()
