from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
import torch
from dpcnn import DPCNN
from csv import reader
import pickle
import torchtext

vocab_size = 802999
label_size = 14

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


model = DPCNN(vocab_size=vocab_size,
             label_size=label_size,
             batchsize=100,
             channels=128,
             embed_dim=128)

model_url = 'https://drive.google.com/u/0/uc?export=download&id=1jzMBAp-qr27-0lIGXzwl0Wsdkpk_4YHX'
checkpoint_file = torchtext.utils.download_from_url(model_url,root='.')

checkpoint = torch.load(checkpoint_file, map_location='cpu')
model.load_state_dict(checkpoint['dpcnn'])

model.eval()

vocab_url = 'https://drive.google.com/u/0/uc?export=download&id=1u1g4-WtR0KIH25AnY30JwBd0-sTRDkHk'
vocab_pickle = torchtext.utils.download_from_url(vocab_url,root='.')
with open(vocab_pickle, 'rb') as handle:
        vocab = pickle.load(handle)

print('Running predictions on 1000 test samples =>')
count = 0
with open('test.csv', 'r') as read_test:
    csv_reader = reader(read_test)
    with open('test.out', 'w', encoding='utf-8') as write_test:
        for row in csv_reader:
                write_test.write(str(predict(row[0],model,vocab,1)))
                count+=1
                if count!=1001:
                    write_test.write('\n')
print('test.out file generated!')

match = 0
count = 0

with open('test.out', 'r') as file1:
    with open('reference.out', 'r') as file2:
        for line1,line2 in zip(file1,file2):
            if line1 == line2:
                match+=1
            count+=1

print('score:%.3f' %(match/count))