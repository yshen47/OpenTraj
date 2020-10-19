import torch
import torch.nn as nn
import random


class Model(nn.Module):
    def __init__(self, context_frame_num, device, hidden_size=64):
        super(Model, self).__init__()
        self.embed = nn.Linear(2, hidden_size)
        self.device = device
        self.epsilon = 0.5
        self.context_frame_num = context_frame_num
        self.GRUCell = torch.nn.GRUCell(hidden_size, hidden_size)
        self.fc = nn.Sequential(*[
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, 2),
        ])

    def forward(self, inputs, context_frame_num):

        batch_size, agent_size, trajectory_length = inputs.shape[:3]
        x = inputs.reshape(-1, *inputs.shape[3:])
        x = self.embed(x)
        x = x.view(batch_size, agent_size, trajectory_length, *x.shape[1:])
        hiddens = []
        res = torch.zeros(*inputs.shape[:2], trajectory_length - self.context_frame_num, 2).to(self.device)
        for time_i in range(trajectory_length):
            for agent_i in range(agent_size):
                if len(hiddens) < agent_size:
                    hidden = self.GRUCell(x[:, agent_i, time_i])
                    hiddens.append(hidden)
                else:
                    if time_i < context_frame_num:
                        hidden = self.GRUCell(x[:, agent_i, time_i], hiddens[agent_i])
                        hiddens[agent_i] = hidden
                    else:
                        if self.epsilon > random.random():
                            hidden = self.GRUCell(hiddens[agent_i], hiddens[agent_i])
                        else:
                            hidden = self.GRUCell(x[:, agent_i, time_i], hiddens[agent_i])
                        curr = self.fc(hidden)
                        res[:, agent_i, time_i - context_frame_num] = curr

        # exclude those preds at moments when inputs doesn't have gt location (out of sight)
        gt_mask = inputs[:, :, context_frame_num:] != 0
        res *= gt_mask
        return res
