# This file should test byol encoder by showing the nearest neighbor
# Load the data and split it into train and test set
import os
import re
import glob
import tqdm
import torch
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

def find_demo_and_demo_index(idx_in_whole_dataset, demo_length_list): # idx_in_whole_dataset is a list
    demo_num_and_demo_index = []
    for idx in idx_in_whole_dataset:
        # print(idx)
        # print(demo_length_list)
        for demo_num, single_demo_length in enumerate(demo_length_list):
            if idx < single_demo_length:
                demo_num_and_demo_index.append((demo_num, idx))
                break
            else:
                idx = idx - single_demo_length
    return demo_num_and_demo_index

def find_nn(repr_idx, repr, test_data, demo_length, demo_dict):
    whole_length = sum(demo_length)
    test_length = len(test_data)
    train_length = whole_length - test_length
    demo_num_list = [key for key in demo_dict.keys()]
    repr_action_id = find_demo_and_demo_index([repr_idx], demo_length)

    # Now find the 10 nn repr in test dataset, and get their indices
    l1_distances = []
    for id, sample in enumerate(test_data):
        l1_distances.append(sample[0].detach().cpu() - repr.detach().cpu())
    l1_distances = np.array(l1_distances)
    l2_distances = np.linalg.norm(l1_distances, axis = 2).squeeze()
    sorted_idxs = np.argsort(l2_distances)
    nn_indices_in_test_dataset = sorted_idxs[:5]
 
    nn_indices_in_whole_dataset = []
    for i in range(len(nn_indices_in_test_dataset)):
        nn_indices_in_whole_dataset.append(nn_indices_in_test_dataset[i] + train_length)
    nn_action_list = find_demo_and_demo_index(nn_indices_in_whole_dataset, demo_length)

    # The indices we are having now starts from the demo clipping point, we need to add that
    for i in range(len(nn_action_list)):
        demo_num, sample_id = nn_action_list[i]
        calibration_timestep_num = demo_dict[demo_num_list[demo_num]]['demo_idx'][0]
        nn_action_list[i] = (demo_num, sample_id + calibration_timestep_num)

    for i in range(len(repr_action_id)):
        demo_num, sample_id = repr_action_id[i]
        calibration_timestep_num = demo_dict[demo_num_list[demo_num]]['demo_idx'][0]
        repr_action_id[i] = (demo_num, sample_id + calibration_timestep_num)

    print('repr_action_id:{}'.format(repr_action_id))
    print('nn_action_list:{}'.format(nn_action_list))
    return repr_action_id, nn_action_list

def find_frame_ids_for_nn(repr_action_id, nn_action_list, data_directory, view_num):
    roots = sorted(glob.glob(f'{data_directory}/demonstration_*'))
    frame_ids = []
    sampled_frames = []
    for root in roots:
        image_idx_file = root + '/image_indices_cam_{}.pkl'.format(view_num)
        with open(image_idx_file, 'rb') as file:
            image_idx = pickle.load(file)
            frame_ids.append(image_idx)

    if repr_action_id != None:
        for i in range(len(repr_action_id)):
            demo_num, action_id = repr_action_id[i]
            image_id = frame_ids[demo_num][action_id][1]
            sampled_frames.append((demo_num, image_id))

    for i in range(len(nn_action_list)):
        demo_num, action_id = nn_action_list[i]
        image_id = frame_ids[demo_num][action_id][1]
        sampled_frames.append((demo_num, image_id))   

    return sampled_frames


def visualization_nn(frame_ids_list, data_directory, view_num, save_dir):
    image_path_list = []
    roots = sorted(glob.glob(f'{data_directory}/demonstration_*'))
    for demo_idx, frame_id in frame_ids_list:
        image_path = roots[demo_idx] + '/cam_{}_rgb_images/frame_{}.png'.format(view_num, str(frame_id).zfill(5))
        image_path_list.append(image_path)
        num_images = len(image_path_list)
        
    num_rows = (num_images + 3) // 6  
    fig, axes = plt.subplots(num_rows, 6, figsize=(16, num_rows * 6))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # # Iterate over each image path and corresponding axis
    for image_path, ax, title in zip(image_path_list, axes.flatten(), frame_ids_list):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis('off')  # Hide axis
        ax.set_title(title) 
        
        plt.tight_layout()
        save_path = save_dir + '/sample_{}_{}.png'.format(demo_idx, frame_id)
        plt.savefig(save_path)
    
        # plt.show()

    return 

def get_deploy_data(deploy_directory, view_num):
    roots = sorted(glob.glob(f'{deploy_directory}/*/1'))
    deployment_length = []
    all_representations = []
    for num in tqdm.trange(len(roots)):
        deployment = roots[num]
        save_path = deployment + '/deployment_byol_representations_cam_{}.pkl'.format(view_num)
        with open(save_path, 'rb') as file:
            representations = pickle.load(file)
        all_representations.append(representations)
        deployment_length.append(len(representations))
    temp = all_representations[0]
    for i in range(1,len(all_representations)):
        temp  = torch.cat((temp, all_representations[i]), dim=0)

    all_representations = temp

    return all_representations, deployment_length



def find_demo_nn(repr, test_data, demo_length, demo_dict):
    demo_num_list = [key for key in demo_dict.keys()]

    # Now find the 10 nn repr in test dataset, and get their indices
    l1_distances = []
    for id, sample in enumerate(test_data):
        l1_distances.append(sample[0].detach().cpu() - repr.detach().cpu())
    l1_distances = np.array(l1_distances)
    l2_distances = np.linalg.norm(l1_distances, axis = 2).squeeze()
    sorted_idxs = np.argsort(l2_distances)
    nn_indices_in_whole_dataset = sorted_idxs[:5]
 
    nn_action_list = find_demo_and_demo_index(nn_indices_in_whole_dataset, demo_length)

    # The indices we are having now starts from the demo clipping point, we need to add that
    for i in range(len(nn_action_list)):
        demo_num, sample_id = nn_action_list[i]
        calibration_timestep_num = demo_dict[demo_num_list[demo_num]]['demo_idx'][0]
        nn_action_list[i] = (demo_num, sample_id + calibration_timestep_num)

    return nn_action_list

def visualization_demo_nn(deploy_ids, frame_ids_list, deploy_directory, data_directory, view_num, save_dir):
    roots = sorted(glob.glob(f'{deploy_directory}/*/1'))
    deploy_image_path = roots[deploy_ids[0]] + '/deployment_image/frame_{}.jpg'.format(str(deploy_ids[1]).zfill(5))

    image_path_list = []
    image_path_list.append(deploy_image_path)
    roots = sorted(glob.glob(f'{data_directory}/demonstration_*'))
    all_ids = [deploy_ids]
    for demo_idx, frame_id in frame_ids_list:
        image_path = roots[demo_idx] + '/cam_{}_rgb_images/frame_{}.png'.format(view_num, str(frame_id).zfill(5))
        image_path_list.append(image_path)
        num_images = len(image_path_list)
        all_ids.append((demo_idx, frame_id))
    print(all_ids)
        
    num_rows = (num_images + 3) // 6  
    fig, axes = plt.subplots(num_rows, 6, figsize=(16, num_rows * 6))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # # Iterate over each image path and corresponding axis
    for image_path, ax, title in zip(image_path_list, axes.flatten(), all_ids):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis('off')  # Hide axis
        ax.set_title(title) 
        
        plt.tight_layout()
        save_path = save_dir + '/sample_{}_{}.png'.format(demo_idx, deploy_ids)
        plt.savefig(save_path)
    
        # plt.show()

    return 


        

        
def main():
    # initialize calibrate_fingertips
    data_directory = Path('/data/irmak/third_person_manipulation/detergent_new_1')
    deploy_directory = Path('/data/irmak/third_person_manipulation/deployments/detergent_new_1')
    view_num = 2
    demo_dict = crop_demo(data_directory, view_num)
    demo_length = get_demo_length(data_directory, view_num)

    find_human_nn_from_deploy = False
    if find_human_nn_from_deploy:
        deploy_representations, deployment_length = get_deploy_data(deploy_directory, view_num)
        _, test_data = get_retargeting_train_val(data_directory, vqbet_get_future_action_chunk = False, view_num = view_num, window_size = 1, action_window_size = 1, train_fraction = 0)
        sampled_indices = random.sample(list(range(sum(deployment_length))), 10)
        for idx in sampled_indices:
            repr = deploy_representations[idx]
            # find the indices of samples from train data that is the nearest neighbor of the given repr
            nn_action_list = find_demo_nn(repr, test_data, demo_length, demo_dict)
            frame_ids_list = find_frame_ids_for_nn(None, nn_action_list, data_directory, view_num) # find the frame ids of the nn for given reprs
            #find the deployment frame ids

            demo_idx = 0
            for length in deployment_length:
                if idx - length >= 0:
                    idx = idx - length
                    demo_idx += 1
                else:
                    deploy_ids = (demo_idx, idx)    
                    break

            visualization_save_dir = '/data/irmak/third_person_manipulation/detergent_new_1'
            visualization_demo_nn(deploy_ids, frame_ids_list, deploy_directory,  data_directory, view_num, visualization_save_dir)
                
    else: 
        train_data, test_data = get_retargeting_train_val(data_directory, vqbet_get_future_action_chunk = False,  view_num = view_num, window_size = 1, action_window_size = 1, train_fraction = 0.1)
        train_length = len(train_data)
        # We dont really need a data loader here
        # randomly sample 10 indices from the train data
        sampled_indices = random.sample(range(train_length), 10)
        for idx in sampled_indices:
            repr, _ = train_data[idx]
            # find the indices of samples from test data that is the nearest neighbor of the given repr
            repr_action_id, nn_action_list = find_nn(idx, repr, test_data, demo_length, demo_dict) # find the action ids of the nn for given reprs
            # here we need image_indices.pkl
            frame_ids_list = find_frame_ids_for_nn(repr_action_id, nn_action_list, data_directory, view_num) # Find the frame ids of these reprs
            visualization_save_dir = '/data/irmak/third_person_manipulation/detergent_new_1'
            visualization_nn(frame_ids_list, data_directory, view_num, visualization_save_dir)



    
if __name__ == '__main__':
    main()
