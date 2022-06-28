import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from .unet import UNet

from torch.utils.tensorboard import SummaryWriter
from .utils.dataset import BasicDataset

from torch.utils.data import DataLoader, random_split
import cv2

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

check_num = 1
lap = 0
##

def make_paths(root_path, check_num, lap, seed):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/likelihoodmap"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/lossmask"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/model_files/model_file-seed{seed:02}"))

    return ps

def mseloss(input, target, mask):
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    ret = (input - target) ** 2
    ret = ret[mask == 1]
    ret = torch.mean(ret)
    return ret

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_net(net,
              device,
              paths,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(paths[0], paths[1], paths[2], img_scale)
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = mseloss
    relu = nn.ReLU()

    with tqdm(total=epochs, leave=False, position=0) as pbar0:
        for epoch in range(epochs):
            pbar0.set_description(f"epoch:{epoch:02}")
            pbar0.update(1)
            net.train()

            epoch_loss = 0
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                lossmask = batch['lossmask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                lossmask = lossmask.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks, lossmask)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar0.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                if global_step % 2 == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        pass
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    pass
                
                if global_step % 50:
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('pred', relu(masks_pred), global_step)
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/lossmask', lossmask, global_step)
                        pass
                    pass
                pass

            if save_cp:
                try:
                    save_path_vector = os.path.join(paths[-1], f"CP_epoch{epoch + 1}.pth")
                    torch.save(net.state_dict(), save_path_vector)
                    pass
                except OSError:
                    pass
                pass
            pass
        pass
        writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=30,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

def main(root_path, check_num, lap, seed):
    paths = make_paths(root_path, check_num, lap, seed)
    ## path list ##
    # 0: cellimage
    # 1: likelihoodmap
    # 2: lossmask
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    set_seed(42+seed)
    # torch.manual_seed(15)
    # np.random.seed(0)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=1, n_classes=1, bilinear=True)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  paths=paths,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=0)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
    main(root_path, check_num, lap, seed)
    print("finish")