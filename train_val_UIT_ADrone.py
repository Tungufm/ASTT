import os
import torch
import torch.nn as nn
import numpy as np
from model import VisionTransformer
# from model_efficientnet import VisionTransformer
import torchvision.transforms as transforms
from config import get_train_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter
from data_utils import DataLoader
from sklearn.metrics import *
import torch.utils.data as data
import pdb
import glob
from torch.autograd import Variable
import argparse
# import neptune

torch.cuda.set_device(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

loss_func_mse = nn.MSELoss(reduction='mean')

def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, device=torch.device('cpu')):
    metrics.reset()
    average_loss = []
    # training loop
    for batch_idx, (batch_data) in enumerate(data_loader):
        batch_data_256, batch_data = batch_data['256'].to(device), batch_data['standard'].to(device)
        optimizer.zero_grad()
        batch_pred = model(batch_data[:,:4])
        loss = loss_func_mse(batch_data_256[:,4].float(), batch_pred)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        average_loss.append(loss.item())

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Reconstruction Loss: {:.4f}"
                    .format(epoch, batch_idx, len(data_loader), np.mean(average_loss)))
    return metrics.result()

def valid_epoch(epoch, model, data_loader, criterion, metrics, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    # validation loop
    new_label = np.load('../../UIT-ADrone/test/test_frame_mask/DJI_0073.npy')

    with torch.no_grad():
        for batch_idx, (batch_data) in enumerate(data_loader):
            batch_data_256, batch_data = batch_data['256'].to(device), batch_data['standard'].to(device)
            # batch_target = batch_target.to(device)
            batch_pred = model(batch_data[:,:4])
            loss = loss_func_mse(batch_data_256[:,4].float(), batch_pred)
            losses.append(loss.item())

    loss = np.mean(losses)
    frame_auc = roc_auc_score(y_true=new_label[:len(losses)], y_score=losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', frame_auc)
    # metrics.update('acc5', acc5)
    print("Test Epoch: {:03d}), AUC@1: {:.2f}".format(epoch, frame_auc))
    return metrics.result()

from tqdm import tqdm
save_path = 'experiments_andt_ADrone_proposed/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

def test_all_scenes(model, test_path, config, device=None):
    path_ckpt = './' + save_path + 'checkpoints/best.pth'
    checkpoint = torch.load(path_ckpt)
    print('Path checkpoint:', path_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    path_scenes = glob.glob(os.path.join(test_path, 'frames/*'))
    path_labels = os.path.join(test_path, 'test_frame_mask/')
    list_np_labels = []

    losses = []
    for idx_video, path_scene in enumerate(path_scenes):
        print('------------------------------------------------------------------------------------------------------')
        print('Number of video:', idx_video+1)
        losses_curr_video = []
        test_dataset = DataLoader(path_scene, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=config.image_size, resize_width=config.image_size)

        test_size = len(test_dataset)
        print('Test size scene ' + path_scene.split('/')[-1] + ': %d' % test_size)
        test_batch = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
        
        np_label = np.load(os.path.join(path_labels, path_scene.split('/')[-1] + '.npy'), allow_pickle=True)
        with torch.no_grad():
            with tqdm(desc='Evaluating ' + path_scene.split('/')[-1], unit='it', total=len(test_batch)) as pbar:
                for batch_idx, (batch_data) in enumerate(test_batch):
                    batch_data_256, batch_data = batch_data['256'].to(device), batch_data['standard'].to(device)
                    batch_pred = model(batch_data[:, :4])
                    loss = loss_func_mse(batch_data_256[:, 4].float(), batch_pred)
                    losses.append(loss.item())
                    losses_curr_video.append(loss.item()) # For visualization
                    pbar.update()

            # list_np_labels.append(np_label[len(np_label) - len(losses_curr_video):])

        np.save(os.path.join(save_path, path_scene.split('/')[-1] + '.npy'), np.array(losses_curr_video))

    # list_np_labels = np.concatenate(list_np_labels)
    loss_all = np.mean(losses)
    print("threshold:", np.mean(losses) + np.std(losses))
    # frame_auc = roc_auc_score(y_true=list_np_labels, y_score=losses)
    # print("Evaluation results:, AUC@1: {:.2f} - Mean loss: {:.2f}".format(frame_auc, loss_all))
    # return frame_auc

def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str('./' + save_path + 'checkpoints/' + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str('./' + save_path + 'checkpoints/' + 'best.pth')
        torch.save(state, filename)

def main():
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    # create model
    print("create model")
    model = VisionTransformer(
             image_size=(config.image_size, config.image_size),
             patch_size=(config.patch_size, config.patch_size),
             emb_dim=config.emb_dim,
             mlp_dim=config.mlp_dim,
             num_heads=config.num_heads,
             num_layers=config.num_layers,
             num_classes=config.num_classes,
             attn_dropout_rate=config.attn_dropout_rate,
             dropout_rate=config.dropout_rate,
             num_frames=config.num_frames)

    # load checkpoint
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path)
        # print(state_dict.keys())
        if config.num_classes != state_dict['classifier.weight'].size(0):
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            del state_dict['transformer.pos_embedding.pos_embedding']
            # del state_dict['transformer.pos_embedding.bias']
            print("re-initialize fc layer")
            print("re-initialize pos_embedding layer")
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(config.checkpoint_path))

    # send model to device
    model = model.to(device)

    if bool(config.train):
        # Loading dataset
        train_folder = "../../UIT-ADrone/train/frames/"
        train_dataset = DataLoader(train_folder, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=config.image_size, resize_width=config.image_size)

        train_size = len(train_dataset)
        print('train size: %d' % train_size)
        train_batch = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                    shuffle=True, num_workers=4, drop_last=True)

        test_folder = "../../UIT-ADrone/test/frames/DJI_0073/"
        test_dataset = DataLoader(test_folder, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=config.image_size, resize_width=config.image_size)

        test_size = len(test_dataset)
        print('test size: %d' % test_size)
        test_batch = data.DataLoader(test_dataset, batch_size=1,
                                    shuffle=False, num_workers=4, drop_last=True)

        print('dataload!')

        # training criterion
        print("create criterion and optimizer")
        criterion = nn.CrossEntropyLoss()

        # create optimizers and learning rate scheduler
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.wd,
            momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config.lr,
            pct_start=config.warmup_steps / config.train_steps,
            total_steps=config.train_steps)

        # start training
        print("start training")
        best_acc = 0.0
        log = {}
        log['val_acc1'] = 0
        # epochs = config.train_steps // len(train_dataloader)
        epochs = config.epochs
        for epoch in range(1, epochs + 1):
            log['epoch'] = epoch

            # train the model
            model.train()
            result = train_epoch(epoch, model, train_batch, criterion, optimizer, lr_scheduler, train_metrics, device)
            log.update(result)

            # validate the model
            if epoch >= 1:
                model.eval()
                result = valid_epoch(epoch, model, test_batch, criterion, valid_metrics, device)
                log.update(**{'val_' + k: v for k, v in result.items()})
            
            # best acc
            best = False
            if log['val_acc1'] > best_acc:
                best_acc = log['val_acc1']
                best = True

            # save model
            save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)

            # print logged informations to the screen
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))
    
    else:
        print('Testing ...')
        test_folder = "../../UIT-ADrone/train/"
        test_all_scenes(model, test_folder, config, device='cuda')

if __name__ == '__main__':
    main()
