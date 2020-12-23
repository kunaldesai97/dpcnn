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
DEBUG = True

# set train parameters
train_params = TrainParameters()
train_params.MAX_EPOCHS = 5
train_params.START_LR = 0.01
train_params.DEV_IDS = [0]
train_params.LOADER_BATCH_SIZE = 80
train_params.LOADER_NUM_THREADS = 0
train_params.VALID_STEPS = 250
train_params.MAX_VALID_BATCHES_NUM = 50
train_params.CHECKPOINT_STEPS = 3000
train_params.VERBOSE_MODE = True
train_params.LOADER_VALID_BATCH_SIZE = train_params.LOADER_BATCH_SIZE+5

# specific unique description for current training experiments
train_params.NAME_TAG = 'dpcnn'
train_params.DESCRIPTION = 'Initial eval'

# [2]
""" Configure ckpt and log directory ---------------------------------------------------------------
"""

checkpoint_dict = {'ckpt': "/scratch/log/Dec06_22-40-41_blu9402u-d01_dpcnn/checkpoints/iter_011001.pth.tar"}
log_dir = '/scratch/log/'
if DEBUG:
    log_dir = None
else:
    print('Log:' + log_dir)

# [3]
""" define dataset ------------------------------------------------------------------------------------------------------
"""
# if os.path.exists('train_data_amazon_full.pickle'):
#     with open('train_data_amazon_full.pickle', 'rb') as handle:
#         train_dataset = pickle.load(handle)
# else:
#     train_dataset, _ = torchtext.datasets.AmazonReviewFull(ngrams=1, root="/scratch/data")
#     with open('train_data_amazon_full.pickle', 'wb') as handle:
#         pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

if os.path.exists('test_data_dbpedia.pickle'):
    with open('test_data_dbpedia.pickle', 'rb') as handle:
        test_dataset = pickle.load(handle)
else:
    _,test_dataset = torchtext.datasets.DBpedia(ngrams=1, root="/scratch/test_data")
    with open('test_data_dbpedia.pickle', 'wb') as handle:
        pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

vocab_len = len(test_dataset.get_vocab())
label_len = len(test_dataset.get_labels())
print("vocab len:", vocab_len, "label_len:", label_len)

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


acc, p, g = train_box.test_loop(test_dataset, generate_batch)
print("test accuracy:", acc)