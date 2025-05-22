import gc
import torch
import argparse
import random
import os
import numpy as np
import csv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dataclasses import dataclass

from preprocessing import *
from dataset import *
from utils import *
from model import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features(model_t, audio_s, video_s, dataloader, save_path):
    """
    从给定的数据加载器中提取隐藏层特征和标签，并保存为 pkl 文件。
    """
    # 切换模型为评估模式
    model_t.eval()
    audio_s.eval()
    video_s.eval()

    all_text_hidden = []
    all_audio_hidden = []
    all_video_hidden = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            # 从 dataloader 中获取数据
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens = batch_input_tokens.cuda()
            attention_masks = attention_masks.cuda()
            audio_inputs = audio_inputs.cuda()
            video_inputs = video_inputs.cuda()
            batch_labels = batch_labels.cuda()

            # 提取隐藏层特征
            text_hidden, _ = model_t(batch_input_tokens, attention_masks)
            audio_hidden, _ = audio_s(audio_inputs)
            video_hidden, _ = video_s(video_inputs)

            # 保存隐藏层特征和对应的标签
            all_text_hidden.append(text_hidden.cpu())
            all_audio_hidden.append(audio_hidden.cpu())
            all_video_hidden.append(video_hidden.cpu())
            all_labels.append(batch_labels.cpu())

    # 将数据拼接成张量
    all_text_hidden = torch.cat(all_text_hidden, dim=0)
    all_audio_hidden = torch.cat(all_audio_hidden, dim=0)
    all_video_hidden = torch.cat(all_video_hidden, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 保存为 pkl 文件
    features_dict = {
        'text_hidden': all_text_hidden,
        'audio_hidden': all_audio_hidden,
        'video_hidden': all_video_hidden,
        'labels': all_labels
    }

    with open(save_path, 'wb') as f:
        pickle.dump(features_dict, f)

    print(f"Features saved to {save_path}")

def extract_all_features(model_t, model_s_a, model_s_v, model_s_a_KD, model_s_v_KD, data_loader, save_path):
    """
    从给定的数据加载器中提取隐藏层特征和标签，并保存为 pkl 文件。
    """
    model_t.eval()
    model_s_a.eval()
    model_s_v.eval()
    model_s_a_KD.eval()
    model_s_v_KD.eval()

    vids = []
    dia2utt = {}
    text = {}
    audio = {}
    video = {}
    audio_kd = {}
    video_kd = {}
    speakers = {}
    labels = {}

    for i, data in enumerate(tqdm(data_loader)):
        # 从 dataloader 中获取数据
        batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels, batch_speakers, batch_dia_ids, batch_utt_ids = data
        batch_input_tokens = batch_input_tokens.cuda()
        attention_masks = attention_masks.cuda()
        audio_inputs = audio_inputs.cuda()
        video_inputs = video_inputs.cuda()
        batch_labels = batch_labels.cuda()
        batch_speakers = batch_speakers.cuda()
        batch_dia_ids = batch_dia_ids.cuda()
        batch_utt_ids = batch_utt_ids.cuda()

        # 提取隐藏层特征
        text_hidden, _ = model_t(batch_input_tokens, attention_masks)
        audio_hidden, _ = model_s_a(audio_inputs)
        video_hidden, _ = model_s_v(video_inputs)
        audio_kd_hidden, _ = model_s_a_KD(audio_inputs)
        video_kd_hidden, _ = model_s_v_KD(video_inputs)

        for i in range(batch_input_tokens.shape[0]):
            vid = batch_dia_ids[i].item()
            if vid not in vids:
                vids.append(batch_dia_ids[i].item())
                dia2utt[vid] = []
                text[vid] = []
                audio[vid] = []
                video[vid] = []
                audio_kd[vid] = []
                video_kd[vid] = []
                speakers[vid] = []
                labels[vid] = []
            dia2utt[vid].append(batch_utt_ids[i].item())
            text[vid].append(text_hidden[i].cpu().numpy())
            audio[vid].append(audio_hidden[i].cpu().numpy())
            video[vid].append(video_hidden[i].cpu().numpy())
            audio_kd[vid].append(audio_kd_hidden[i].cpu().numpy())
            video_kd[vid].append(video_kd_hidden[i].cpu().numpy())
            speakers[vid].append(batch_speakers[i].item())
            labels[vid].append(batch_labels[i].item())

    features_dict = {
        'text': text,
        'audio': audio,
        'video': video,
        'audio_kd': audio_kd,
        'video_kd': video_kd,
        'speakers': speakers,
        'labels': labels,
        'vids': vids,
        'dia2utt': dia2utt
    }

    with open(save_path, 'wb') as f:
        pickle.dump(features_dict, f)

    print(f"Features saved to {save_path}")






if __name__ == '__main__':
    # release gpu memory
    gc.collect()
    torch.cuda.empty_cache()

    # setting args
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', type=int, default=10, help='epoch for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate for training.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training.')
    parser.add_argument('--seed', type=int, default=2024, help='random seed for training.')
    parser.add_argument('--train', type=bool, default=True, help='whether to train the model.')
    parser.add_argument('--teacher', type=str, default='text', help='who is teacher?')
    parser.add_argument('--student', type=str, default='text', help='who is student?')
    parser.add_argument('--advanced', type=bool, default=False, help='whether to use advanced loss.')
    args = parser.parse_args()

    # set seed
    seed_everything(args.seed)

    @dataclass
    class Config():
        mask_time_length: int = 3

    # modal_name
    text_model = "roberta-large"
    audio_model = "data2vec-audio-base-960h"
    video_model = "timesformer-base-finetuned-k400"

    # load data
    data_path = "../datasets/MELD"
    train_path = os.path.join(data_path, "train_meld_emo.csv")
    dev_path = os.path.join(data_path, "dev_meld_emo.csv")
    test_path = os.path.join(data_path, "test_meld_emo.csv")

    train_dataset = MELD_Dataset(preprocessing(train_path, split_type='train'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=SequentialSampler(train_dataset), num_workers=16, collate_fn=all_features_batchs)

    dev_dataset = MELD_Dataset(preprocessing(dev_path, split_type='dev'))
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, sampler=SequentialSampler(dev_dataset), num_workers=16, collate_fn=all_features_batchs)

    test_dataset = MELD_Dataset(preprocessing(test_path, split_type='test'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=SequentialSampler(test_dataset), num_workers=16, collate_fn=all_features_batchs)

    save_path = os.path.join('./MELD/save_model')
    print("###Save Path### ", save_path)

    clsNum = len(train_dataset.emoList)
    init_config = Config()

    model_t = Text_model(text_model, clsNum)
    model_t.load_state_dict(torch.load(os.path.join(save_path, 'text.bin')))
    model_s_a = Audio_model(audio_model, clsNum, init_config)
    model_s_a.load_state_dict(torch.load(os.path.join(save_path, 'audio.bin')))
    model_s_v = Video_model(video_model, clsNum)
    model_s_v.load_state_dict(torch.load(os.path.join(save_path, 'video.bin')))

    model_s_a_KD = Audio_model(audio_model, clsNum, init_config)
    model_s_a_KD.load_state_dict(torch.load(os.path.join(save_path, 'text_KD_audio.bin')))
    model_s_v_KD = Video_model(video_model, clsNum)
    model_s_v_KD.load_state_dict(torch.load(os.path.join(save_path, 'text_KD_video.bin')))


    for para in model_t.parameters():
        para.requires_grad = False

    for para in model_s_a.parameters():
        para.requires_grad = False

    for para in model_s_v.parameters():
        para.requires_grad = False

    for para in model_s_a_KD.parameters():
        para.requires_grad = False

    for para in model_s_v_KD.parameters():
        para.requires_grad = False

    model_t = model_t.cuda()
    model_t.eval()

    model_s_a = model_s_a.cuda()
    model_s_a.eval()

    model_s_v = model_s_v.cuda()
    model_s_v.eval()

    model_s_a_KD = model_s_a_KD.cuda()
    model_s_a_KD.eval()

    model_s_v_KD = model_s_v_KD.cuda()
    model_s_v_KD.eval()

    save_path = "feature/first_stage_train_features.pkl"
    extract_all_features(model_t, model_s_a, model_s_v, model_s_a_KD, model_s_v_KD, train_loader, save_path)

    save_path = "feature/first_stage_dev_features.pkl"
    extract_all_features(model_t, model_s_a, model_s_v, model_s_a_KD, model_s_v_KD, dev_loader, save_path)

    save_path = "feature/first_stage_test_features.pkl"
    extract_all_features(model_t, model_s_a, model_s_v, model_s_a_KD, model_s_v_KD, test_loader, save_path)

    print("---------------Done--------------")



