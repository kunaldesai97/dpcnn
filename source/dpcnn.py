import pickle
import torch
import torchtext
from torchtext.datasets import text_classification
from torch.utils.data import DataLoader
from os import path
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import math
# import torch_xla
# import torch_xla.core.xla_model as xm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DPCNN_rep_block(torch.nn.Module):
    def __init__(self, batchsize, channels=250):
        """
        vocab_size, to create embedding layer
        text_length, to create fixed length samples
        """
        super(DPCNN_rep_block, self).__init__()
        self.channels = channels
        self.batchsize = batchsize

        self.conv1 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1)
        self.conv2 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1)
        self.downsample = torch.nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.downsample(x)

        # pre-activation
        sc = x
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = x + sc.expand(x.shape)

        return x


class DPCNN(torch.nn.Module):

    def __init__(self, vocab_size, label_size, batchsize, channels, embed_dim):
        """
        vocab_size, to create embedding layer
        text_length, to create fixed length samples
        """
        super(DPCNN, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.channels = channels
        # self.text_length = text_length
        self.batchsize = batchsize
        self.embed_dim = embed_dim

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.region_embedding = nn.Sequential(
            nn.Conv1d(1, self.channels, kernel_size=3, padding=1))
        # self.region_embedding = nn.Conv2d(1, self.channels, (3, self.embed_dim), stride=1)
        self.pre_conv1 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1)
        self.pre_conv2 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1)


        self.block1 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block2 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block3 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block4 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block5 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block6 = DPCNN_rep_block(self.batchsize, self.channels)

        self.final_pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = torch.nn.Dropout(p = .5)
        self.fc = torch.nn.Linear(self.channels, self.label_size)

    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
        x.unsqueeze_(1)
        x = self.region_embedding(x)

        # pre-activation
        sc = x
        x = self.pre_conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pre_conv2(x)
        x = torch.nn.functional.relu(x)
        x = x + sc.expand(x.shape)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        # x = torch.nn.functional.log_softmax(x, dim=1)

        return x
