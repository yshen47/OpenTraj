import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Model(nn.Module):
    def __init__(self, hidden_size=128):
        super(Model, self).__init__()
        self.embed = nn.Linear(4, hidden_size)
        self.epsilon = 0.5
        self.GRUCell = torch.nn.GRUCell(hidden_size, hidden_size)
        self.fc = nn.Sequential(*[
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, 2),
        ])

    def forward(self, pos_x, pos_y, v_x, v_y, context_frame_num):
        res = []
        x = torch.stack([pos_x, pos_y, v_x, v_y]).permute(0, 2, 1)
        batch_size, trajectory_length = x.shape[:2]
        x = x.reshape(-1, *x.shape[2:])
        x = self.embed(x)
        x = x.view(batch_size, trajectory_length, *x.shape[1:])
        hidden = None
        for i in range(v_x.shape[1]):
            if hidden is None:
                hidden = self.GRUCell(x[:, i])
            else:
                if i < context_frame_num:
                    hidden = self.GRUCell(x[:, i], hidden)
                else:
                    if self.epsilon > random.random():
                        hidden = self.GRUCell(hidden, hidden)
                    else:
                        hidden = self.GRUCell(x[:, i], hidden)
            curr = self.fc(hidden)
            res.append(curr)
        res = torch.stack(res).permute(1,0,2)
        return res[:, :, 0], res[:, :, 1]