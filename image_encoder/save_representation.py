# save the representation of images from dataset, 
# in case we don't want to train the encoder together with vae
# 1. load the trained encoder
# 2. Apply encoder and save representations
import re
import os
import cv2
import glob
import tqdm
import torch
import hydra
import pickle
from pathlib import Path

from PIL import Image as im
from omegaconf import OmegaConf
import torchvision.transforms as T
from collections import OrderedDict
from third_person_man.utils import VISION_IMAGE_MEANS, VISION_IMAGE_STDS, crop_transform

def init_encoder_info(out_dir, device):
    cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
    image_encoder_path = os.path.join(out_dir, 'models/byol_encoder_best.pt')
    image_encoder = load_model(cfg, device, image_encoder_path, model_type='image')
    return image_encoder 

def load_model(cfg, device, model_path, model_type=None):
    # Initialize the model
    if cfg.learner_type == 'bc':
        if model_type == 'image':
            model = hydra.utils.instantiate(cfg.encoder.image_encoder)
        elif model_type == 'tactile':
            model = hydra.utils.instantiate(cfg.encoder.tactile_encoder)
        elif model_type == 'last_layer':
            model = hydra.utils.instantiate(cfg.encoder.last_layer)
    elif cfg.learner_type == 'bc_gmm':
        model = hydra.utils.instantiate(cfg.learner.gmm_layer)
    elif cfg.learner_type == 'temporal_ssl':
        if model_type == 'image':
            model = hydra.utils.instantiate(cfg.encoder.encoder)
        elif model_type == 'linear_layer':
            model = hydra.utils.instantiate(cfg.encoder.linear_layer)
    else:
        model = hydra.utils.instantiate(cfg.model)  

    state_dict = torch.load(model_path) # All the parameters by default gets installed to cuda 0
    
    # Modify the state dict accordingly - this is needed when multi GPU saving was done
    if cfg.distributed:
        state_dict = modify_multi_gpu_state_dict(state_dict)
    
    if 'byol' in cfg.learner_type or 'vicreg' in cfg.learner_type:
        state_dict = modify_byol_state_dict(state_dict)

    # Load the new state dict to the model 
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

def modify_multi_gpu_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

def modify_byol_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if '.net.' in k:
            name = k.split('.net.')[-1] # This is hard coded fixes for vicreg
            new_state_dict[name] = v
    return new_state_dict


def dump_video_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame
        frame_path = f"{output_folder}/frame_{str(frame_count).zfill(5)}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()

    return



def save_image_representations(data_path, out_dir, device, view_num):
    # Load encoder
    image_transform = T.Compose([
        T.Resize((480,640)),
        T.Lambda(crop_transform),
        T.ToTensor(),
        T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])
    image_encoder = init_encoder_info(out_dir, device)
    roots = sorted(glob.glob(f'{data_path}/demonstration_*'))
    print(roots)
    
    for num in tqdm.trange(len(roots)):
        # if num != (len(roots) - 1):
        #     continue 

        demo = roots[num]
        demo_representations = []
        save_path = demo + '/img_byol_representations_cam_{}.pkl'.format(view_num)
        if os.path.exists(save_path):
            os.remove(save_path)

        image_idx_file = Path(demo +  '/image_indices_cam_{}.pkl'.format(view_num))
        with open(image_idx_file, 'rb') as file:
            image_indices = pickle.load(file)
        
            for num in tqdm.trange(len(image_indices)):
                idx = image_indices[num]
                image_path = os.path.join(demo, 'cam_{}_rgb_images/frame_{}.png'.format(view_num, str(idx[1]).zfill(5)))
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = im.fromarray(image)
                image = torch.FloatTensor(image_transform(image)).to(device)
                image = image_encoder(image.unsqueeze(dim=0)).detach().cpu()
                demo_representations.append(image)

            demo_representations = torch.stack(demo_representations, dim=0)
            with open(save_path, 'wb') as file:
                pickle.dump(demo_representations, file)
            torch.cuda.empty_cache()
    return

def save_deployment_representations(deployment_path, out_dir, device, view_num):
    # Load encoder
    image_transform = T.Compose([
        T.Resize((480,640)),
        T.Lambda(crop_transform),
        T.ToTensor(),
        T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS),
        ])
    image_encoder = init_encoder_info(out_dir, device)

    
    roots = sorted(glob.glob(f'{deployment_path}/*/1'))
    print(roots)
    
    for root in roots:
        # First dump the images of videos if the images are not yet dumped
        if os.path.exists(root + '/deployment_image'):
            print('deployment images already exists!!!!')
            continue
        else:
            video = root + '/deployment_w_axes.mp4'
            # create the video path
            os.makedirs(root + '/deployment_image')
            dump_video_frames(video, root + '/deployment_image')
    
    for num in tqdm.trange(len(roots)):
        demo = roots[num]
        demo_representations = []
        save_path = demo + '/deployment_byol_representations_cam_{}.pkl'.format(view_num)
        if os.path.exists(save_path):
            os.remove(save_path)

        # get all the frames
        image_list = sorted(glob.glob(f'{demo}/deployment_image/frame_*'))

        for idx in tqdm.trange(len(image_list)):
            image_path = image_list[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = im.fromarray(image)
            image = torch.FloatTensor(image_transform(image)).to(device)
            image = image_encoder(image.unsqueeze(dim=0)).detach().cpu()
            demo_representations.append(image)

        demo_representations = torch.stack(demo_representations, dim=0)
        with open(save_path, 'wb') as file:
            pickle.dump(demo_representations, file)
        torch.cuda.empty_cache()
    return

    
if __name__ == '__main__':
    out_dir = Path('/home/irmak/Workspace/third-person-manipulation/out/2024.05.21/18-32_detergent_new_1_byol_seed_5')
    device = torch.device('cuda:0')
    data_path = Path('/data/irmak/third_person_manipulation/detergent_new_1')
    view_num =2

    save_deployment = True 
    deployment_path = Path('/data/irmak/third_person_manipulation/deployments/detergent_new_1')
    save_image_representations(data_path, out_dir, device, view_num)
    if save_deployment: 
        save_deployment_representations(deployment_path,out_dir, device, view_num )



