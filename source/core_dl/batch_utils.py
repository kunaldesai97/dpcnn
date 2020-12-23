import torch

def my_collate(batch):
    data = [item[1] for item in batch]
    target = [item[0] for item in batch]
    target = torch.LongTensor(target)
    return [target, data]


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label