import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle


class IEMOCAPDataset(Dataset):
    def __init__(self, data):
        self.emoList = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        self.session_dataset = data

    def __len__(self):
        return len(self.session_dataset)

    def __getitem__(self, idx):
        return self.session_dataset[idx]


class AudioDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.dialogue_ids = self.data['dialogue_id']
        self.utterance_ids = self.data['utterance_id']
        self.features = self.data['features']
        self.labels = self.data['label']
        self.emoList = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        self.label_to_index = {label: idx for idx, label in enumerate(self.emoList)}

    def load_data(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        label_index = self.label_to_index[label]
        return [feature, label_index]


class IEMOCAP_Dataset(Dataset):
    def __init__(self, data_path):
        self.emoList = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        data = pickle.load(open(data_path, 'rb'))
        self.text = data['text']
        self.audio = data['audio']
        self.video = data['video']
        self.audio_kd = data['audio_kd']
        self.video_kd = data['video_kd']
        self.speakers = data['speakers']
        self.labels = data['labels']
        self.vids = data['vids']
        self.dia2utt = data['dia2utt']
        self.len = len(self.vids)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        vid = self.vids[idx]
        return torch.FloatTensor(np.array(self.text[vid])),\
               torch.FloatTensor(np.array(self.audio[vid])),\
               torch.FloatTensor(np.array(self.video[vid])),\
               torch.FloatTensor(np.array(self.audio_kd[vid])),\
               torch.FloatTensor(np.array(self.video_kd[vid])),\
               torch.FloatTensor(np.array(self.speakers[vid])),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               vid

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<5 else pad_sequence(dat[i], True) if i<8 else dat[i].tolist() for i in dat]

