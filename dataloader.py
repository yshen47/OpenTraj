import torch
import os
import numpy as np
from toolkit.loaders.loader_eth import load_eth
import random


class ETHDataset(torch.utils.data.Dataset):

    def __init__(self, mode, trajectory_length):
        """

        :type mode: str ['train', 'val', 'test']
        """
        self.mode = mode
        self.trajectory_length = trajectory_length
        annot_file = os.path.join('datasets', 'ETH/seq_eth/obsmat.txt')
        self.traj_dataset = load_eth(annot_file)
        agent_ids = np.unique(self.traj_dataset.data['agent_id'])
        random.seed(1)
        np.random.shuffle(agent_ids)
        filtered_agent_ids = []
        for agent_id in agent_ids:
            indices = np.where(self.traj_dataset.data['agent_id'].to_numpy() == agent_id)[0]
            if len(indices) > trajectory_length:
                filtered_agent_ids.append(agent_id)
        if mode == 'train':
            self.agent_ids = filtered_agent_ids[:int(len(filtered_agent_ids) * 0.8)]
        elif mode == 'val':
            self.agent_ids = filtered_agent_ids[int(len(filtered_agent_ids) * 0.8):int(len(filtered_agent_ids) * 0.9)]
        else:
            self.agent_ids = filtered_agent_ids[int(len(filtered_agent_ids) * 0.9):]

    def __len__(self):
        return len(self.agent_ids)

    def __getitem__(self, item):
        agent_id = self.agent_ids[item]
        indices = np.where(self.traj_dataset.data['agent_id'].to_numpy() == agent_id)[0]
        pos_x = self.traj_dataset.data.pos_x.to_numpy()[indices]
        pos_y = self.traj_dataset.data.pos_y.to_numpy()[indices]
        v_x = self.traj_dataset.data.vel_x.to_numpy()[indices]
        v_y = self.traj_dataset.data.vel_y.to_numpy()[indices]

        start_index = random.randint(0, len(v_x) - self.trajectory_length - 1)
        pos_x = torch.tensor(pos_x, dtype=torch.float)[start_index:start_index+self.trajectory_length]
        pos_y = torch.tensor(pos_y, dtype=torch.float)[start_index:start_index+self.trajectory_length]
        v_x = torch.tensor(v_x, dtype=torch.float)[start_index:start_index+self.trajectory_length]
        v_y = torch.tensor(v_y, dtype=torch.float)[start_index:start_index+self.trajectory_length]
        return pos_x, pos_y, v_x, v_y