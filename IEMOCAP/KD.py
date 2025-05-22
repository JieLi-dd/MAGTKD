import gc
import torch
import argparse
import random
import os
import numpy as np
import csv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from dataclasses import dataclass
from scipy.stats import wasserstein_distance

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


def wasserstein_distance_loss(a, b):
    # 计算每一对样本的Wasserstein距离
    return wasserstein_distance(a, b)


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

class Logit_Loss(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=2.0):
        super(Logit_Loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss

class Feature_Loss(nn.Module):
    def __init__(self, temp=1.0):
        super(Feature_Loss, self).__init__()
        self.t = temp

    def forward(self, other_embd, text_embd):
        text_embd = F.normalize(text_embd, p=2, dim=1)
        other_embd = F.normalize(other_embd, p=2, dim=1)
        target = torch.matmul(text_embd, text_embd.transpose(0,1))
        x = torch.matmul(text_embd, other_embd.transpose(0,1))
        log_q = torch.log_softmax(x / self.t, dim=1)
        p = torch.softmax(target / self.t, dim=1)
        return F.kl_div(log_q, p, reduction='batchmean')

class Feature_Loss_advanced(nn.Module):
    def __init__(self, temp=1.0, alpha=0.5):
        super(Feature_Loss_advanced, self).__init__()
        self.t = temp
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Balances between similarity and uniqueness

    def forward(self, other_embd, text_embd):
        # Normalize embeddings
        text_embd = F.normalize(text_embd, p=2, dim=1)
        other_embd = F.normalize(other_embd, p=2, dim=1)
        original_other_embd = F.normalize(other_embd, p=2, dim=1)

        # Calculate similarity loss (distillation)
        target = torch.matmul(text_embd, text_embd.transpose(0, 1))

        x = torch.matmul(text_embd, other_embd.transpose(0, 1))
        log_q = torch.log_softmax(x / self.t, dim=1)
        p = torch.softmax(target / self.t, dim=1)
        similarity_loss = F.kl_div(log_q, p, reduction='batchmean')

        # Calculate uniqueness loss (preserve original audio characteristics)
        uniqueness_loss = F.mse_loss(other_embd, original_other_embd)

        # Combine both losses
        return self.alpha * similarity_loss + (1 - self.alpha) * uniqueness_loss

def CE_Loss(args, pred_outs, logit_t, hidden_s, hidden_t, labels):
    ori_loss = nn.CrossEntropyLoss()
    ori_loss = ori_loss(pred_outs, labels)
    logit_loss = Logit_Loss().cuda()
    logit_loss = logit_loss(pred_outs, logit_t)
    feature_loss = Feature_Loss().cuda()
    # if args.advanced:
    #     feature_loss = Feature_Loss_advanced().cuda()
    feature_loss = feature_loss(hidden_s, hidden_t)

    # loss_val = ori_loss + 0.1*logit_loss + feature_loss
    return ori_loss + logit_loss


def CELoss(pred_logits, labels):
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_logits, labels)
    return loss_val

def model_train(args, epochs, model_t, model_s, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, scaler, save_path, model_name):
    best_dev_fscore, best_test_fscore = 0, 0

    model_t.eval()
    for epoch in tqdm(range(epochs)):
        model_s.train()
        for i_batch, data in enumerate(train_loader):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                """Predication"""
                batch_text_tokens, batch_attention_masks, batch_audios, batch_videos, batch_labels = data
                batch_text_tokens, batch_attention_masks, batch_audios, batch_videos, batch_labels = batch_text_tokens.cuda(), batch_attention_masks.cuda(), batch_audios.cuda(), batch_videos.cuda(), batch_labels.cuda()
                if args.student == 'text':
                    hidden_s, logit_s = model_s(batch_text_tokens, batch_attention_masks)
                elif args.student == 'audio':
                    hidden_s, logit_s = model_s(batch_audios)
                elif args.student == 'video':
                    hidden_s, logit_s = model_s(batch_videos)

                if args.teacher == 'text':
                    hidden_t, logit_t = model_t(batch_text_tokens, batch_attention_masks)
                elif args.teacher == 'audio':
                    hidden_t, logit_t = model_t(batch_audios)
                elif args.teacher == 'video':
                    hidden_t, logit_t = model_t(batch_videos)
                loss_val = CE_Loss(args, logit_s, logit_t, hidden_s, hidden_t, batch_labels)

            scaler.scale(loss_val).backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        model_s.eval()
        dev_pred_list, dev_label_list = evalution(args, model_s, dev_loader)
        dev_pre, dev_rec, dev_f1, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        dev_acc = accuracy_score(dev_label_list, dev_pred_list)
        print(f"dev_acc: {dev_acc}; dev_fscore: {dev_f1}\n")

        if dev_f1 > best_dev_fscore:
            best_dev_fscore = dev_f1
            _SaveModel(model_s, save_path, model_name)

            model_s.eval()
            test_pred_list, test_label_list = evalution(args, model_s, test_loader)
            test_pre, test_rec, test_f1, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
            test_acc = accuracy_score(test_label_list, test_pred_list)
            print(f"test_acc: {test_acc}; test_fscore: {test_f1}\n")
            print(classification_report(test_label_list, test_pred_list, digits=4))


def evalution(args, model_s, dataloader):
    model_s.eval()
    label_list = []
    pred_list = []
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            """Prediction"""
            batch_text_tokens, batch_attention_masks, batch_audios, batch_videos, batch_labels = data
            batch_text_tokens, batch_attention_masks, batch_audios, batch_videos, batch_labels = batch_text_tokens.cuda(), batch_attention_masks.cuda(), batch_audios.cuda(), batch_videos.cuda(), batch_labels.cuda()
            if args.student == 'text':
                hidden_s, logit_s = model_s(batch_text_tokens, batch_attention_masks)
            elif args.student == 'audio':
                hidden_s, logit_s = model_s(batch_audios)
            elif args.student == 'video':
                hidden_s, logit_s = model_s(batch_videos)

            """Calculation"""
            pred_label = logit_s.argmax(dim=1).detach().cpu().numpy()
            true_label = batch_labels.detach().cpu().numpy()

            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, save_path, model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, model_name))


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
    parser.add_argument('--student', type=str, default='audio', help='who is student?')
    parser.add_argument('--advanced', type=bool, default=True, help='whether to use advanced loss.')
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
    data_path = "../datasets/IEMOCAP"
    train_path = os.path.join(data_path, "IEMOCAP_train.csv")
    dev_path = os.path.join(data_path, "IEMOCAP_dev.csv")
    test_path = os.path.join(data_path, "IEMOCAP_test.csv")

    train_dataset = IEMOCAPDataset(preprocessing(train_path, split_type='train'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=all_batchs)

    dev_dataset = IEMOCAPDataset(preprocessing(dev_path, split_type='dev'))
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=all_batchs)

    test_dataset = IEMOCAPDataset(preprocessing(test_path, split_type='test'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=all_batchs)

    save_path = os.path.join('./IEMOCAP/save_model')
    if args.advanced:
        model_name = args.teacher + '_KD_' + args.student + "_advanced" + '.bin'
    else:
        model_name = args.teacher + '_KD_' + args.student + '.bin'
    print("###Save Path### ", save_path)

    clsNum = len(train_dataset.emoList)
    init_config = Config()
    if args.train:
        if args.teacher == 'text':
            model_t = Text_model(text_model, clsNum)
            model_t.load_state_dict(torch.load(os.path.join(save_path, 'text.bin')))
        elif args.teacher == 'audio':
            model_t = Audio_model(audio_model, clsNum, init_config)
            model_t.load_state_dict(torch.load(os.path.join(save_path, 'audio.bin')))
        elif args.teacher == 'video':
            model_t = Video_model(video_model, clsNum)
            model_t.load_state_dict(torch.load(os.path.join(save_path, 'video.bin')))
        else:
            print("No such teacher!")

        if args.student == 'text':
            model_s = Text_model(text_model, clsNum)
            model_s.load_state_dict(torch.load(os.path.join(save_path, 'text.bin')))
        elif args.student == 'audio':
            model_s = Audio_model(audio_model, clsNum, init_config)
            model_s.load_state_dict(torch.load(os.path.join(save_path, 'audio.bin')))
        elif args.student == 'video':
            model_s = Video_model(video_model, clsNum)
            # model_s.load_state_dict(torch.load(os.path.join(save_path, 'video.bin')))
        else:
            print("No such student!")
    else:
        if args.teacher == 'text':
            model_t = Text_model(text_model, clsNum)
            model_t.load_state_dict(torch.load(os.path.join(save_path, 'text.bin')))
        elif args.teacher == 'audio':
            model_t = Audio_model(audio_model, clsNum, init_config)
            model_t.load_state_dict(torch.load(os.path.join(save_path, 'audio.bin')))
        elif args.teacher == 'video':
            model_t = Video_model(video_model, clsNum)
            model_t.load_state_dict(torch.load(os.path.join(save_path, 'video.bin')))
        else:
            print("No such teacher!")

        if args.student == 'text':
            model_s = Text_model(text_model, clsNum)
            model_s.load_state_dict(torch.load(os.path.join(save_path, model_name)))
        elif args.student == 'audio':
            model_s = Audio_model(audio_model, clsNum, init_config)
            model_s.load_state_dict(torch.load(os.path.join(save_path, model_name)))
        elif args.student == 'video':
            model_s = Video_model(video_model, clsNum)
            model_s.load_state_dict(torch.load(os.path.join(save_path, model_name)))
        else:
            print("No such student!")

    for para in model_t.parameters():
        para.requires_grad = False

    model_t = model_t.cuda()
    model_t.eval()

    model_s = model_s.cuda()
    model_s.eval()

    """Training Setting"""
    training_epochs = args.epochs
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset) * training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model_s.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    if args.train:
        model_train(args, training_epochs, model_t, model_s, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, scaler, save_path, model_name)
    else:
        test_pred_list, test_label_list = evalution(args, model_s, test_loader)
        test_pre, test_rec, test_f1, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
        test_acc = accuracy_score(test_label_list, test_pred_list)
        print(f"test_acc: {test_acc}; test_fscore: {test_f1}\n")
    print("---------------Done--------------")



