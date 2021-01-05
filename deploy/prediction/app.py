# Serve model as a flask application

import pickle
# import numpy as np
from flask import Flask, request
import torchtext
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
import torch
from dpcnn import DPCNN
from service import PredictionService
from flask import Blueprint
import sys
import requests
import logging
import simplejson as json
import urllib
from flask import Response

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model = None
vocab = None

app = Flask(__name__)


bp = Blueprint('app', __name__)

# def predict(text, model, vocab, ngrams):
#     tokenizer = get_tokenizer("basic_english")
#     with torch.no_grad():
#         text = torch.tensor([vocab[token]
#                             for token in ngrams_iterator(tokenizer(text), ngrams)])
#         output = model(text, torch.tensor([0]))
#         return output.argmax(1).item() + 1

def load():

    global model
    global vocab

    vocab_size = 802999
    label_size = 14

    model = DPCNN(vocab_size=vocab_size,
             label_size=label_size,
             batchsize=100,
             channels=128,
             embed_dim=128)

    # model variable refers to the global variable
    # model_url = 'https://drive.google.com/u/0/uc?export=download&id=1jzMBAp-qr27-0lIGXzwl0Wsdkpk_4YHX'
    # checkpoint_file = torchtext.utils.download_from_url(model_url,root='.')
    checkpoint_file = 'dbpedia_model.pth.tar'

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint['dpcnn'])

    model.eval()

    # vocab_url = 'https://drive.google.com/u/0/uc?export=download&id=1u1g4-WtR0KIH25AnY30JwBd0-sTRDkHk'
    # vocab_pickle = torchtext.utils.download_from_url(vocab_url,root='.')
    with open('vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

@bp.route('/health')
def health():
    return Response("", status=200, mimetype="application/json")

@bp.route('/readiness')
def readiness():
    return Response("", status=200, mimetype="application/json")


@bp.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        try:
            content = request.get_json()  # Get data posted as a json
            text = content['input']
        except: 
            return json.dumps({"message": "Error reading arguments"})
    return str(PredictionService.predict(text,model,vocab,1))

app.register_blueprint(bp, url_prefix='/api/v1/model/')

if __name__ == '__main__':
    load()  # load model and vocabulary at the beginning once only
    if len(sys.argv) < 2:
        logging.error("missing port arg 1")
        sys.exit(-1)

    p = int(sys.argv[1])
    app.run(host='0.0.0.0', port=p, threaded=True)