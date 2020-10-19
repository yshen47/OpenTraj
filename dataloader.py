import torch
import os
import numpy as np
from toolkit.loaders.loader_eth import load_eth
import random


class ETHDataset(torch.utils.data.Dataset):

    def __init__(self, mode, trajectory_length, context_length, agent_buffer_size):
        """
        mode:                       str         ['train', 'val', 'test']
        trajectory_length:          int         time window for a batch data, each frame is 0.4 sec
        context_length:             int         known past trajectories
        agent_buffer_size:          int         max number of agents allowed in one data item
        """
        self.mode = mode
        assert trajectory_length > context_length
        self.trajectory_length = trajectory_length
        self.agent_buffer_size = agent_buffer_size
        self.context_length = context_length
        annot_file = os.path.join('datasets', 'ETH/seq_eth/obsmat.txt')
        traj_dataset = load_eth(annot_file)
        total = traj_dataset.data['timestamp'].last_valid_index()
        total_time = traj_dataset.data['timestamp'][total] - traj_dataset.data['timestamp'][0]
        # split datasets based on ratio of total time
        # checked labels for the dataset. all are pedestrians.
        if mode == 'train':
            start_time = traj_dataset.data['timestamp'][0]
            end_time = traj_dataset.data['timestamp'][0] + total_time * 0.8
        elif mode == 'val':
            start_time = traj_dataset.data['timestamp'][0] + total_time * 0.8
            end_time = traj_dataset.data['timestamp'][0] + total_time * 0.9
        else:
            start_time = traj_dataset.data['timestamp'][0] + total_time * 0.8
            end_time = traj_dataset.data['timestamp'][0] + total_time
        self.traj_dataset = self.extract_dataset(traj_dataset, start_time, end_time)

    def extract_dataset(self, dataset, start_time, end_time):
        valid_indices = np.where((dataset.data['timestamp'] >= start_time) & (dataset.data['timestamp'] < end_time))
        dataset.data = dataset.data.iloc[valid_indices]
        return dataset

    def __len__(self):
        return len(self.traj_dataset.data) - self.trajectory_length

    def __getitem__(self, item):
        start_index = item
        subset = self.traj_dataset.data.iloc[start_index:start_index+self.trajectory_length]

        start_timestamp = subset['timestamp'].iloc[0]
        agents = list(set(subset['agent_id'].iloc[:self.context_length])) # to ensure that agents appear in context frames, otherwise it's unreasonable to predict its future occurence.
        random.shuffle(agents)
        agents = agents[:self.agent_buffer_size]

        # group trajectory by agents
        agent_trajs = torch.zeros([self.agent_buffer_size, self.trajectory_length,2]) # (agent_dim, temporal_dim, pos)

        for i in range(self.trajectory_length):
            curr_time_id = int((subset['timestamp'].iloc[i] - start_timestamp) / 0.4)
            if subset['agent_id'].iloc[i] in agents and curr_time_id < self.trajectory_length:
                curr_agent_id = agents.index(subset['agent_id'].iloc[i])
                curr_pos_x = subset['pos_x'].iloc[i]
                curr_pos_y = subset['pos_y'].iloc[i]
                agent_trajs[curr_agent_id][curr_time_id][0] = curr_pos_x
                agent_trajs[curr_agent_id][curr_time_id][1] = curr_pos_y
        return agent_trajs