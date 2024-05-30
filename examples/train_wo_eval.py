import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

# import kitchen_env
import wandb
from video import VideoRecorder
import pickle

config_name = "train_retargeting"

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="configs", config_name=config_name, version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    train_data, test_data = hydra.utils.instantiate(cfg.data)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, pin_memory=False
    )
    if "visual_input" in cfg and cfg.visual_input:
        print("use visual environment")
        cfg.model.gpt_model.config.input_dim = 1024
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )

    for data in test_data:
        continue

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = run.name or "Offline"
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)
    video = VideoRecorder(dir_name=save_path)



    for epoch in tqdm.trange(cfg.epochs):
        cbet_model.eval()

        if epoch % cfg.eval_freq == 0:
            total_loss = 0
            action_diff = 0
            action_diff_tot = 0
            action_diff_mean_res1 = 0
            action_diff_mean_res2 = 0
            action_diff_max = 0
            with torch.no_grad():
                for data in test_loader:
                    # obs, act, goal = (x.to(cfg.device) for x in data)
                    # predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
                    obs, act = (x.to(cfg.device) for x in data)
                    print('shape of sampled observation: {}'.format(obs.shape))
                    # bet model should accept None goal as long as goal_dim is set to 0 
                    predicted_act, loss, loss_dict = cbet_model(obs, None, act)
                    total_loss += loss.item()
                    wandb.log({"eval/{}".format(x): y for (x, y) in loss_dict.items()})
                    action_diff += loss_dict["action_diff"]
                    action_diff_tot += loss_dict["action_diff_tot"]
                    action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
                    action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
                    action_diff_max += loss_dict["action_diff_max"]
            print(f"Test loss: {total_loss / len(test_loader)}")
            wandb.log({"eval/epoch_wise_action_diff": action_diff})
            wandb.log({"eval/epoch_wise_action_diff_tot": action_diff_tot})
            wandb.log({"eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1})
            wandb.log({"eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2})
            wandb.log({"eval/epoch_wise_action_diff_max": action_diff_max})

        for data in tqdm.tqdm(train_loader):
            if epoch < (cfg.epochs * 0.5):
                optimizer["optimizer1"].zero_grad()
                optimizer["optimizer2"].zero_grad()
            else:
                optimizer["optimizer2"].zero_grad()
            # obs, act, goal = (x.to(cfg.device) for x in data)
            # predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
            obs, act = (x.to(cfg.device) for x in data)
            # bet model should accept None goal as long as goal_dim is set to 0 
            predicted_act, loss, loss_dict = cbet_model(obs, None, act)
            wandb.log({"train/{}".format(x): y for (x, y) in loss_dict.items()})
            loss.backward()
            if epoch < (cfg.epochs * 0.5):
                optimizer["optimizer1"].step()
                optimizer["optimizer2"].step()
            else:
                optimizer["optimizer2"].step()

        if epoch % cfg.save_every == 0:
            cbet_model.save_model(save_path)

    return 


if __name__ == "__main__":
    main()
