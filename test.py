import datetime
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report

def test(model, test_dataloader, args, classes_name=None):
    test_dataloader = tqdm(test_dataloader, total=len(test_dataloader), ncols=80, leave=False,
                           desc=colored('Testing on Test', 'yellow'))

    model.eval()
    with torch.no_grad():
        prediction = []
        prediction_score = []
        true_label = []
        for i, batch in enumerate(test_dataloader):
            X, y = batch
            logits, info_loss  = model(X)#, _  新增占位
            pred = torch.argmax(logits, dim=-1)
            prediction.extend(pred.detach().cpu().tolist())
            true_label.extend(y.detach().cpu().tolist())
            prediction_score.extend(logits.detach().cpu().tolist())

        prediction = np.array(prediction, dtype=np.int64)
        true_label = np.array(true_label, dtype=np.int64)
        prediction_score = np.array(prediction_score)
        prediction_score = prediction_score / prediction_score.sum(axis=1, keepdims=True)

        acc = accuracy_score(true_label, prediction)
        f1 = f1_score(true_label, prediction, average="macro", zero_division=0)
        mcc = matthews_corrcoef(true_label, prediction)
        kappa = cohen_kappa_score(true_label, prediction)
        auc = roc_auc_score(true_label, prediction_score, multi_class="ovo")

        cm = confusion_matrix(true_label, prediction)
        report = classification_report(true_label, prediction)

    return acc, f1, mcc, kappa, auc, cm, report
