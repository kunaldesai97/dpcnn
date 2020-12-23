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
tqdm = partial(tqdm, position=0)
# device = xm.xla_device()

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


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

    def __init__(self, vocab_size, label_size, text_length, batchsize, channels, embed_dim):
        """
        vocab_size, to create embedding layer
        text_length, to create fixed length samples
        """
        super(DPCNN, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.channels = channels
        self.text_length = text_length
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

        # self.rep_blocks = nn.Sequential(
        #     DPCNN_rep_block(self.batchsize, self.channels),
        #     DPCNN_rep_block(self.batchsize, self.channels),
        #     DPCNN_rep_block(self.batchsize, self.channels),
        #     DPCNN_rep_block(self.batchsize, self.channels),
        #     DPCNN_rep_block(self.batchsize, self.channels),
        #     DPCNN_rep_block(self.batchsize, self.channels),
        # )

        self.block1 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block2 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block3 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block4 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block5 = DPCNN_rep_block(self.batchsize, self.channels)
        self.block6 = DPCNN_rep_block(self.batchsize, self.channels)

        self.final_pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = torch.nn.Dropout(p = .5)
        self.fc = torch.nn.Linear(self.channels, self.label_size)

    # Region Embedding function
    # TODO: should length be fixed somewhere else?
    # TODO: take in batch instead of sample
    def region_emb(self, tokens):

        # create embedding
        res = np.zeros(self.vocab_size)
        for i in range(len(tokens) - 1):
            vec = np.zeros(self.vocab_size)
            vec[tokens[i].item()] = 1
            vec[tokens[i + 1].item()] = 1
            res = np.vstack((res, vec))

        # enforce fixed length samples
        if res.shape[0] >= self.text_length + 1:
            return torch.from_numpy(res[1:self.text_length + 1]).float()
        else:
            ext = np.zeros(self.vocab_size)
            while res.shape[0] != self.text_length + 1:
                res = np.vstack((res, ext))
            return torch.from_numpy(res[1:]).float()

    def region_emb_batch(self, tokens):

        fin = torch.zeros(len(tokens), self.text_length, self.vocab_size)
        for b in range(len(tokens)):
            token = tokens[b]
            # create embedding
            res = np.zeros(self.vocab_size)
            for i in range(min(len(token) - 1, self.text_length)):
                vec = np.zeros(self.vocab_size)
                vec[token[i].item()] = 1
                vec[token[i + 1].item()] = 1
                res = np.vstack((res, vec))

            # enforce fixed length samples
            if res.shape[0] >= self.text_length + 1:
                fin[b] = torch.from_numpy(res[1:self.text_length + 1]).float()
            else:
                ext = np.zeros(self.vocab_size)
                while res.shape[0] != self.text_length + 1:
                    res = np.vstack((res, ext))
                fin[b] = torch.from_numpy(res[1:]).float()
        return fin

    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
        x.unsqueeze_(1)
        x = self.region_embedding(x)
        # print(x.size())
        # exit()
        # exit()
        # x.unsqueeze_(0)
        # batch = tokens.shape[0]
        # batch = len(tokens)
        # region embedding
        # x = self.region_emb(tokens)
        # x = self.region_emb_batch(tokens)
        # x = x.resize(self.batchsize, 1, self.text_length, self.vocab_size)

        # pre-activation
        sc = x
        x = self.pre_conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pre_conv2(x)
        x = torch.nn.functional.relu(x)
        x = x + sc.expand(x.shape)
        # print(x.size())
        # exit()

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # x = self.rep_blocks(x)

        x = self.final_pool(x)
        # print("AAAAAAAAAAAAAAA",x.size())
        # exit()
        x = x.view(x.size(0), -1)
        # print(x.size())
        # exit()
        x = self.dropout(x)
        x = self.fc(x)

        # x = torch.nn.functional.log_softmax(x, dim=1)

        return x


if __name__ == '__main__':

    # training hyper parameters
    epochs = 30
    sample_length = 50
    batchsize = 100  # need dataloader to change
    embed_dim = 32

    # model hyper parameters
    channels = 2

    # get data
    if False and path.exists('train_data.pickle') and path.exists('test_data.pickle'):
        with open('train_data.pickle', 'rb') as handle:
            train_dataset = pickle.load(handle)
        with open('test_data.pickle', 'rb') as handle:
            test_dataset = pickle.load(handle)

    else:
        train_dataset, test_dataset = torchtext.datasets.DBpedia(ngrams=1, root="/content/drive/MyDrive/data")
        with open('train_data.pickle', 'wb') as handle:
            pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('test_data.pickle', 'wb') as handle:
            pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_subset, valid_subset = torch.utils.data.random_split(train_dataset,
                                                               [train_dataset.__len__() - 10000, 10000,],
                                                               generator=torch.Generator().manual_seed(42))


    # data = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,
    #                   collate_fn=generate_batch)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,
                                                      [math.floor(train_dataset.__len__()*0.8),
                                                       train_dataset.__len__() - math.floor(train_dataset.__len__()*0.8)],
                                                      generator=torch.Generator().manual_seed(42))


    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,
                      collate_fn=generate_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=True,
                      collate_fn=generate_batch)

    # prep data
    # TODO: create fixed length data, currently done in region embedding...

    # initilize model
    dpcnn = DPCNN(vocab_size=vocab_len, label_size=label_len, text_length=sample_length, batchsize=batchsize,
                  channels=channels, embed_dim = embed_dim).to(device)

    # # create data loaders
    # # TODO: batch in data loader

    # train
    # for epoch in range(epochs):
    #     for label, tokens in train_dataset:
    #         output = dpcnn(tokens)
    #         print(output.shape)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(dpcnn.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 24, gamma=0.009)

    track_trainloss = {}
    track_validloss = {}


    for epoch in tqdm(range(epochs)):
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        # epoch_num += 1
        for train_i, (text,offsets,cls) in enumerate(tqdm(train_dataloader)):
            dpcnn.train()
            optimizer.zero_grad()
            text = text.to(device)
            offsets = offsets.to(device)
            cls = cls.to(device)
            output = dpcnn(text, offsets)
            # prediction = torch.max(output,1)[1]
            loss = criterion(output,cls)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # train_acc += (output.argmax(1) == cls).sum().item()

        with torch.no_grad():
            for valid_i, (text,offsets,cls) in enumerate(tqdm(valid_dataloader)):
                text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
                dpcnn.eval()
                output = dpcnn(text, offsets)
                loss = criterion(output,cls)
                valid_loss += loss.item()


        scheduler.step()
        track_trainloss[epoch] = train_loss/(train_i+1)
        track_validloss[epoch] = valid_loss/(valid_i+1)
        print('\nepoch:%d, train loss:%.3f, valid loss:%.3f' %(epoch, train_loss/(train_i+1), valid_loss/(valid_i+1)))

    with open('trainloss.pickle', 'wb') as handle:
        pickle.dump(track_trainloss, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open('validloss.pickle', 'wb') as handle:
        pickle.dump(track_validloss, handle, protocol=pickle.HIGHEST_PROTOCOL)


    torch.save(dpcnn.state_dict(), 'model30e.pt')



            # print(output.shape)
            # print(output[0])
            # exit()
            # print(cls)
            # prediction = torch.max(output,1)[1]
            # print(prediction)
            # exit()
            # with torch.no_grad():


