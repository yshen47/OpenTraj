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
@click.option('--experiment_name', type=str, default="none")
@click.option('--steps', type=int, default=0)
@click.option('--num_workers', type=int, default=1)
@click.option('--last_checkpoint', type=str, default=None)
@click.option('--batch_size', type=int, default=4)
@click.option('--epoch_num', type=int, default=100)
@click.option('--lr', type=float, default=1e-3)
@click.option('--context_frame_num', type=int, default=8)
@click.option('--trajectory_interval', type=int, default=20) # each frame is 0.4 sec. It seems it takes 10 sec on average to walk on the trajectory
@click.option('--agent_buffer_size', type=int, default=8)
def main(experiment_name, steps, num_workers, last_checkpoint, batch_size, epoch_num, lr, context_frame_num, trajectory_interval, agent_buffer_size):
    model_id = datetime.now().strftime("%m-%d-%H:%M:%S")
    model_dir = os.path.join("trained_models", experiment_name, model_id)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tb_logger.configure(model_dir, flush_secs=5)

    # torch.backends.cudnn.benchmark = True
    best_val_loss = np.Inf
    device = torch.device('cuda:0')

    train_dataset = ETHDataset("train", trajectory_interval, context_frame_num, agent_buffer_size)
    train_data_loader = data.DataLoader(train_dataset, drop_last=True, shuffle=True,
                                        num_workers=num_workers, pin_memory=True, batch_size=batch_size)

    val_dataset = ETHDataset("val", trajectory_interval, context_frame_num, agent_buffer_size)
    val_data_loader = data.DataLoader(val_dataset, batch_size=batch_size, drop_last=False,
                                      num_workers=num_workers, pin_memory=True, shuffle=False)

    model = Model(context_frame_num, device).to(device)
    if last_checkpoint is not None:
        saved_model = torch.load(os.path.join(last_checkpoint, 'model_best.pth.tar'))
        model.load_state_dict(saved_model['model'])
        best_val_loss = saved_model['valid_loss']
        steps = saved_model["steps"]
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    criterion = torch.nn.MSELoss().to(device)
    for e in range(epoch_num):
        model.train()
        print("Epoch: ", str(e))
        for i, inputs in tqdm(
                enumerate(train_data_loader), total=len(train_data_loader)):
            inputs = inputs.to(device)
            preds = model(inputs, context_frame_num)
            loss = criterion(preds, inputs[:, :, context_frame_num:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('train_loss: {}'.format(loss.item()))
            tb_logger.log_value("train_loss", loss.item())
            steps += 1

        model.eval()
        val_loss = 0.0
        for i, inputs in tqdm(
                enumerate(val_data_loader), total=len(val_data_loader)):
            inputs = inputs.to(device)
            preds = model(inputs, context_frame_num)
            loss = criterion(preds, inputs[:, :, context_frame_num:])
            tb_logger.log_value("val_loss", loss.item())
            val_loss += loss.item()
        val_loss /= len(val_data_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'model': model.state_dict(),
                'valid_loss': best_val_loss,
                'steps': steps,
            }, prefix=model_dir)
        else:
            save_checkpoint({
                'model': model.state_dict(),
                'valid_loss': best_val_loss,
                'steps': steps,
            }, prefix=model_dir + '/checkpoint_{}.pth.tar'.format(steps))


if __name__ == '__main__':
    main()
