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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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

def CELoss(pred_logits, labels):
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_logits, labels)
    return loss_val

def model_train(epochs, model, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, save_path):
    best_dev_fscore, best_test_fscore = 0, 0

    for epochs in tqdm(range(epochs)):
        model.train()
        for i_batch, data in enumerate(train_loader):
            optimizer.zero_grad()

            """Predication"""
            batch_text_tokens, batch_attention_masks, batch_labels = data
            batch_text_tokens, batch_attention_masks, batch_labels = batch_text_tokens.cuda(), batch_attention_masks.cuda(), batch_labels.cuda()
            last_hidden, pred_logits = model(batch_text_tokens, batch_attention_masks)

            loss_val = CELoss(pred_logits, batch_labels)
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

        model.eval()
        dev_pred_list, dev_label_list = evalution(model, dev_loader)
        dev_pre, dev_rec, dev_f1, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        dev_acc = accuracy_score(dev_label_list, dev_pred_list)
        print(f"dev_acc: {dev_acc}; dev_fscore: {dev_f1}\n")

        if dev_f1 > best_dev_fscore:
            best_dev_fscore = dev_f1
            _SaveModel(model, save_path)

            model.eval()
            test_pred_list, test_label_list = evalution(model, test_loader)
            test_pre, test_rec, test_f1, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
            test_acc = accuracy_score(test_label_list, test_pred_list)
            print(f"test_acc: {test_acc}; test_fscore: {test_f1}\n")


def evalution(model, dataloader):
    model.eval()
    label_list = []
    pred_list = []
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            """Prediction"""
            batch_text_tokens, batch_attention_masks, batch_labels = data
            batch_text_tokens, batch_attention_masks, batch_labels = batch_text_tokens.cuda(), batch_attention_masks.cuda(), batch_labels.cuda()
            last_hidden, pred_logits = model(batch_text_tokens, batch_attention_masks)

            """Calculation"""
            pred_label = pred_logits.argmax(dim=1).detach().cpu().numpy()
            true_label = batch_labels.detach().cpu().numpy()

            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'text.bin'))


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
    args = parser.parse_args()

    # set seed
    seed_everything(args.seed)

    # load data
    text_model = "roberta-large"

    data_path = "../datasets/IEMOCAP"
    train_path = os.path.join(data_path, "IEMOCAP_train.csv")
    dev_path = os.path.join(data_path, "IEMOCAP_dev.csv")
    test_path = os.path.join(data_path, "IEMOCAP_test.csv")

    train_dataset = IEMOCAPDataset(preprocessing(train_path, split_type='train'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=text_batchs)

    dev_dataset = IEMOCAPDataset(preprocessing(dev_path, split_type='dev'))
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=text_batchs)

    test_dataset = IEMOCAPDataset(preprocessing(test_path, split_type='test'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=text_batchs)

    save_path = os.path.join('./IEMOCAP/save_model')
    print("###Save Path### ", save_path)

    clsNum = len(train_dataset.emoList)
    if args.train:
        model = Text_model(text_model, clsNum)
    else:
        model = Text_model(text_model, clsNum)
        model.load_state_dict(torch.load(os.path.join(save_path, 'text.bin')))
    model = model.cuda()
    model.eval()

    """Training Setting"""
    training_epochs = args.epochs
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset) * training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    if args.train:
        model_train(training_epochs, model, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, save_path)
    else:
        test_pred_list, test_label_list = evalution(model, test_loader)
        test_pre, test_rec, test_f1, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
        test_acc = accuracy_score(test_label_list, test_pred_list)
        print(f"test_acc: {test_acc}; test_fscore: {test_f1}\n")
    print("---------------Done--------------")



