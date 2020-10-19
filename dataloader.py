import torch
import os
import numpy as np
from toolkit.loaders.loader_eth import load_eth
import random
import pandas as pd


class ETHDataset(torch.utils.data.Dataset):

    def __init__(self, mode, trajectory_interval, context_length, agent_buffer_size):
        """
        mode:                       str         ['train', 'val', 'test']
        trajectory_interval:        int         time window for a batch data, each frame is 0.4 sec
        context_length:             int         known past trajectories
        agent_buffer_size:          int         max number of agents allowed in one data item
        """
        self.mode = mode
        assert trajectory_interval > context_length
        self.trajectory_interval = trajectory_interval
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
        return len(self.traj_dataset.data) - self.trajectory_interval

    @staticmethod
    def count_trailing_zero(l):
        c = 0
        for a in reversed(l):
            if a[0] == 0 and a[0] == 0:
                c += 1
            else:
                break
        return c

    def __getitem__(self, item):
        start_index = item
        start_timestamp = self.traj_dataset.data.timestamp[start_index]
        end_timestamp = start_timestamp + self.trajectory_interval * 0.4
        indices = np.where((self.traj_dataset.data.timestamp >= start_timestamp) & (self.traj_dataset.data.timestamp < end_timestamp))[0]
        subset = self.traj_dataset.data.iloc[indices]
        agents = list(set(subset['agent_id'].iloc[:self.context_length])) # to ensure that agents appear in context frames, otherwise it's unreasonable to predict its future occurence.
        random.shuffle(agents)
        agents = agents[:self.agent_buffer_size]

        # group trajectory by agents
        agent_trajs = np.zeros([self.agent_buffer_size, self.trajectory_interval, 2]) # (agent_dim, temporal_dim, pos)

        for i in range(subset.shape[0]):
            curr_time_id = int((subset['timestamp'].iloc[i] - start_timestamp) / 0.4)
            if subset['agent_id'].iloc[i] in agents and curr_time_id < self.trajectory_interval:
                curr_agent_id = agents.index(subset['agent_id'].iloc[i])
                curr_pos_x = subset['pos_x'].iloc[i]
                curr_pos_y = subset['pos_y'].iloc[i]
                agent_trajs[curr_agent_id][curr_time_id][0] = curr_pos_x
                agent_trajs[curr_agent_id][curr_time_id][1] = curr_pos_y

        # linear interpolation to fill missing data in each trajectory
        for i, agent_traj in enumerate(agent_trajs):
            df = pd.DataFrame(data=agent_traj, columns=['pos_x', 'pos_y'])
            trailing_zero = ETHDataset.count_trailing_zero(agent_traj)
            df = df.replace(0, np.nan)
            pos_x = df['pos_x'].interpolate().to_numpy()
            pos_y = df['pos_y'].interpolate().to_numpy()
            if trailing_zero > 0:
                pos_x[-trailing_zero:] = 0
                pos_y[-trailing_zero:] = 0
            agent_trajs[i, :, 0] = pos_x
            agent_trajs[i, :, 1] = pos_y
        agent_trajs = np.nan_to_num(agent_trajs)
        if np.isnan(agent_trajs).any():
            print('here')
            pass
        agent_trajs = torch.tensor(agent_trajs, dtype=torch.float)
        return agent_trajs