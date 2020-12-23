import os, shutil, warnings, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.utils.data import dataset, dataloader
from core_dl.train_params import TrainParameters
from core_dl.base_train_box import BaseTrainBox
from core_dl.torch_vision_ext import *
from dpcnn import DPCNN


class DPCNNTrainBox(BaseTrainBox):

    def __init__(self, train_params: TrainParameters,
                 vocab_size, label_size, text_length, batchsize, embed_dim,
                 log_dir=None,
                 ckpt_path_dict=None,  # {'ckpt': ckpt_path (optional)}
                channels=250):

        assert ckpt_path_dict is not None
        self.vocab_size, self.label_size, self.text_length, self.batchsize, self.channels, self.embed_dim\
            = vocab_size, label_size, text_length, batchsize, channels, embed_dim
        super(DPCNNTrainBox, self).__init__(train_params, log_dir,
                                                     checkpoint_path=ckpt_path_dict[
                                                         'ckpt'] if 'ckpt' in ckpt_path_dict else None,
                                                     comment_msg=train_params.NAME_TAG,
                                                     load_optimizer=True)

    def _set_loss_func(self):
        self.criterion = nn.CrossEntropyLoss()

    def _set_optimizer(self):
        # config the optimizer
        super(DPCNNTrainBox, self)._set_optimizer()
        self.optimizer = torch.optim.RMSprop([
            {'params': self.dpcnn.parameters(), 'lr': self.train_params.START_LR}], lr=self.train_params.START_LR)

        def accuracy(probs, target):
            winners = probs.argmax(dim=1).view(-1)
            corrects = (winners == target)
            return corrects.sum().float() / float(target.size(0))

        self.acc = accuracy


    def _set_network(self):
        super(DPCNNTrainBox, self)._set_network()
        with torch.cuda.device(self.dev_ids[0]):
            self.dpcnn = DPCNN(vocab_size=self.vocab_size, label_size=self.label_size, text_length=self.text_length,
                               batchsize=self.batchsize,
                               channels=self.channels,
                               embed_dim=self.embed_dim)
            self.dpcnn.cuda()

    def _load_network_from_ckpt(self, checkpoint_dict):
        # load network from checkpoint, ignore the instance if not found in dict.
        super(DPCNNTrainBox, self)._load_network_from_ckpt(checkpoint_dict)
        with torch.cuda.device(self.dev_ids[0]):
            if 'dpcnn' in checkpoint_dict:
                self.dpcnn.load_state_dict(checkpoint_dict['dpcnn'])
                self.dpcnn.cuda()

    def _save_checkpoint_dict(self, checkpoint_dict: dict):
        # save the instance when save_check_point was activated in training loop
        super(DPCNNTrainBox, self)._save_checkpoint_dict(checkpoint_dict)
        checkpoint_dict['dpcnn'] = self.dpcnn.state_dict()

    """ Train Routines -------------------------------------------------------------------------------------------------
        """

    def _prepare_train(self):
        self.dpcnn.train()

    def _train_feed(self, train_sample, cur_train_epoch, cur_train_itr, eval_flag=False) -> dict():

        super(DPCNNTrainBox, self)._train_feed(train_sample, cur_train_epoch, cur_train_itr)
        with torch.cuda.device(self.dev_ids[0]):
            cur_dev = torch.cuda.current_device()
            opt = self.dpcnn(train_sample[0].cuda(), train_sample[1].cuda())
            loss = self.criterion(opt, train_sample[2].reshape(-1).cuda())
            loss.backward()
            self.optimizer.step()

            acc = self.acc(opt, train_sample[2].reshape(-1).cuda()).item()
            # print(loss)
            return {'Loss(Train)/batch_loss': loss,
                    'Accuracy(Train)/batch_acc': acc}

    """ Validation Routines --------------------------------------------------------------------------------------------
        """

    def _prepare_eval(self):
        self.dpcnn.eval()

    def _valid_loop(self, valid_loader, cur_train_epoch, cur_train_itr):
        super(DPCNNTrainBox, self)._valid_loop(valid_loader, cur_train_epoch, cur_train_itr)
        accs = []
        losses = []
        for valid_batch_idx, valid_sample in enumerate(valid_loader):
            with torch.cuda.device(self.dev_ids[0]):
                cur_dev = torch.cuda.current_device()
                opt = self.dpcnn(valid_sample[0].cuda(), valid_sample[1].cuda())
                losses.append(self.criterion(opt, valid_sample[2].reshape(-1).cuda()).item())
                accs.append(self.acc(opt, valid_sample[2].cuda()).item())
        return {'Loss(Valid)/batch_loss': np.mean(losses),
                'Accuracy(Valid)/batch_acc': np.mean(accs)}

    def test_loop(self, valid_data, collate_fn, shuffle=False, batch_size=100):
        # super(DPCNNTrainBox, self).test_loop(valid_data, shuffle, batch_size, max_test_itr)

        if valid_data is not None and isinstance(valid_data, dataset.Dataset):
            valid_loader = dataloader.DataLoader(valid_data,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 pin_memory=self.train_params.LOADER_PIN_MEM,
                                                 num_workers=self.train_params.LOADER_NUM_THREADS,
                                                 drop_last=True,
                                                 collate_fn=collate_fn)
            if self.train_params.VERBOSE_MODE:
                print("Test set: %d items" % (len(valid_data)))

        elif valid_data is not None and isinstance(valid_data, dataloader.DataLoader):
            valid_loader = valid_data
        else:
            valid_loader = None

        self.dpcnn.eval()

        gt = np.array([])
        pred = np.array([])

        for valid_batch_idx, valid_sample in enumerate(valid_loader):
            with torch.cuda.device(self.dev_ids[0]):
                cur_dev = torch.cuda.current_device()
                opt = self.dpcnn(valid_sample[0].cuda(), valid_sample[1].cuda())
                pred = np.concatenate((pred, opt.argmax(dim=1).reshape(-1).cpu().numpy()))
                gt = np.concatenate((gt, valid_sample[2].reshape(-1).cpu().numpy()))
        corrects = (pred == gt)
        return (np.sum(corrects) / float(len(gt))), pred, gt