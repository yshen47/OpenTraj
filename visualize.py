from dataloader import ETHDataset
import click
import tensorboard_logger as tb_logger
from torch.utils import data
from tqdm import tqdm
from utils.helper import save_checkpoint
import torch
from model import Model
import numpy as np
import os
from datetime import datetime

@click.command()
@click.option('--steps', type=int, default=0)
@click.option('--num_workers', type=int, default=4)
@click.option('--last_checkpoint', type=str, default=os.path.join('trained_models', 'naive_lstm', '10-25-20:28:45'))
@click.option('--batch_size', type=int, default=4)
@click.option('--context_frame_num', type=int, default=8)
@click.option('--trajectory_interval', type=int, default=20) # each frame is 0.4 sec. It seems it takes 10 sec on average to walk on the trajectory
@click.option('--agent_buffer_size', type=int, default=8)
def main(steps, num_workers, last_checkpoint, batch_size, context_frame_num, trajectory_interval, agent_buffer_size):

    # torch.backends.cudnn.benchmark = True
    best_val_loss = np.Inf
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = ETHDataset("test", trajectory_interval, context_frame_num, agent_buffer_size)
    test_data_loader = data.DataLoader(test_dataset, batch_size=batch_size, drop_last=False,
                                      num_workers=num_workers, pin_memory=True, shuffle=False)

    model = Model(context_frame_num, device).to(device)
    if last_checkpoint is not None:
        saved_model = torch.load(os.path.join(last_checkpoint, 'model_best.pth.tar'))
        model.load_state_dict(saved_model['model'])
        best_val_loss = saved_model['valid_loss']
        steps = saved_model["steps"]
    criterion = torch.nn.MSELoss().to(device)

    model.eval()
    test_loss = 0.0
    for i, inputs in tqdm(
            enumerate(test_data_loader), total=len(test_data_loader)):
        inputs = inputs.to(device)
        preds = model(inputs, context_frame_num)
        loss = criterion(preds, inputs[:, :, context_frame_num:])
        test_loss += loss.item()
    test_loss /= len(test_data_loader)
    print("test_ADE: {}".format(test_loss))


if __name__ == '__main__':
    main()
