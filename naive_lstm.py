from dataloader import ETHDataset
import torch
import click
import numpy as np
import os
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
@click.option('--experiment_name', type=str, default="train_seg_bdd_mix")
@click.option('--steps', type=int, default=0)
@click.option('--num_workers', type=int, default=1)
@click.option('--last_checkpoint', type=str, default=None)
@click.option('--batch_size', type=int, default=4)
@click.option('--epoch_num', type=int, default=100)
@click.option('--lr', type=float, default=1e-4)
@click.option('--context_frame_num', type=int, default=8)
@click.option('--trajectory_length', type=int, default=16)
def main(experiment_name, steps, num_workers, last_checkpoint, batch_size, epoch_num, lr, context_frame_num, trajectory_length):
    model_id = datetime.now().strftime("%m-%d-%H:%M:%S")
    model_dir = os.path.join("trained_models", experiment_name, model_id)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tb_logger.configure(model_dir, flush_secs=5)

    # torch.backends.cudnn.benchmark = True
    best_val_loss = np.Inf
    device = torch.device('cuda:0')

    train_dataset = ETHDataset("train", trajectory_length)
    train_data_loader = data.DataLoader(train_dataset, drop_last=True, shuffle=False,
                                        num_workers=num_workers, pin_memory=True, batch_size=batch_size)

    val_dataset = ETHDataset("val", trajectory_length)
    val_data_loader = data.DataLoader(val_dataset, batch_size=batch_size, drop_last=False,
                                      num_workers=num_workers, pin_memory=True, shuffle=False)

    model = Model().to(device)
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
        for i, (pos_x, pos_y, v_x, v_y) in tqdm(
                enumerate(train_data_loader), total=len(train_data_loader)):
            pos_x = pos_x.to(device)
            pos_y = pos_y.to(device)
            v_x = v_x.to(device)
            v_y = v_y.to(device)
            pred_x, pred_y = model(pos_x, pos_y, v_x, v_y, context_frame_num)
            #print(pred_x, pred_y)
            print(pos_x, pos_y)
            loss_x = criterion(pred_x[:, context_frame_num:], v_x[:, context_frame_num:])
            loss_y = criterion(pred_y[:, context_frame_num:], v_y[:, context_frame_num:])
            loss = loss_x + loss_y
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('train_loss: {}'.format(loss.item()))
if __name__ == '__main__':
    main()