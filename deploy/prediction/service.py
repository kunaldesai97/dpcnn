from GoogleNews import GoogleNews
from model import Model
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#1. Return the list of links+label
#2. Store it in the DB

class PredictionService:

    global labels 
    
    labels = {"1":"Company",
              "2":"EducationalInstitution",
              "3":"Artist",
              "4":"Athlete",
              "5":"OfficeHolder",
              "6":"MeanOfTransportation",
              "7":"Building",
              "8":"NaturalPlace",
              "9":"Village",
              "10":"Animal",
              "11":"Plant",
              "12":"Album",
              "13":"Film",
              "14":"WrittenWork"}

    def predict(text, model, vocab, ngrams):
        tokenizer = get_tokenizer("basic_english")
        with torch.no_grad():
            text_token = torch.tensor([vocab[token]
                                for token in ngrams_iterator(tokenizer(text), ngrams)])
            output = model(text_token, torch.tensor([0]))
            class_label = labels[str(output.argmax(1).item() + 1)]
            googlenews = GoogleNews()
            googlenews.get_news(class_label) # replace with the value of output
            links = googlenews.get_links()
            out = Model.addOutput(links,class_label,text)
            return (class_label, out)