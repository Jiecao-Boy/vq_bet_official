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

        demo = roots[num]
        demo_representations = []
        save_path = demo + '/img_representations_cam_{}.pkl'.format(view_num)
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

if __name__ == '__main__':
    out_dir = Path('/home/irmak/Workspace/third-person-manipulation/out/2024.04.25/14-10_mustard_picking_byol_seed_5')
    device = torch.device('cuda:0')
    data_path = Path('/data/irmak/third_person_manipulation/mustard_picking')
    view_num =1
    save_image_representations(data_path, out_dir, device, view_num)



