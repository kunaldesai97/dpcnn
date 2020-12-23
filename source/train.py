import os, shutil
from core_dl.train_params import TrainParameters
import pickle
import torch
import torchtext
from trainbox import DPCNNTrainBox
import numpy as np


# [1]
""" Train Parameters ---------------------------------------------------------------------------------------------------
"""
# toggle `DEBUG` to disable logger (won't dump to disk)
DEBUG = False

# set train parameters
train_params = TrainParameters()
train_params.MAX_EPOCHS = 10
train_params.START_LR = 0.01
train_params.DEV_IDS = [0]
train_params.LOADER_BATCH_SIZE = 100
train_params.LOADER_NUM_THREADS = 0
train_params.VALID_STEPS = 250
train_params.MAX_VALID_BATCHES_NUM = 50
train_params.CHECKPOINT_STEPS = 3000
train_params.VERBOSE_MODE = True
train_params.LOADER_VALID_BATCH_SIZE = train_params.LOADER_BATCH_SIZE
train_params.LR_DECAY_FACTOR = 0.1
train_params.LR_DECAY_STEPS = 8


# specific unique description for current training experiments
train_params.NAME_TAG = 'dpcnn'
train_params.DESCRIPTION = 'Initial eval'

# [2]
""" Configure ckpt and log directory ---------------------------------------------------------------
"""

checkpoint_dict = {'ckpt': None}
log_dir = './log/'
if DEBUG:
    log_dir = None
else:
    print('Log:' + log_dir)

# [3]
""" define dataset ------------------------------------------------------------------------------------------------------
"""

if os.path.exists('train_data.pickle'):
    with open('train_data.pickle', 'rb') as handle:
        train_dataset = pickle.load(handle)
else:
    train_dataset, _ = torchtext.datasets.DBpedia(ngrams=1, root="data")
    with open('train_data.pickle', 'wb') as handle:
        pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

vocab_len = len(train_dataset.get_vocab())
label_len = len(train_dataset.get_labels())
print("vocab len:", vocab_len, "label_len:", label_len)

train_dset, valid_dset = torch.utils.data.random_split(train_dataset,
                                                               [train_dataset.__len__() - 10000, 10000,])

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

# [4]
""" Train --------------------------------------------------------------------------------------------------------------
"""

# training hyper parameters
batchsize = train_params.LOADER_BATCH_SIZE

# model hyper parameters
channels = 128
embed_dim = 128

train_box = DPCNNTrainBox(train_params=train_params,
                            vocab_size=vocab_len, label_size=label_len,
                            batchsize=batchsize,
                            log_dir=log_dir,
                            ckpt_path_dict=checkpoint_dict,
                            channels=channels,
                            embed_dim=embed_dim
                          )


if not DEBUG:
    shutil.copy(os.path.realpath(__file__), train_box.model_def_dir)  # save the train interface to model def dir
train_box.train_loop(train_dset, collate_fn=generate_batch, valid_data=valid_dset)