# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
import pandas as pd
import nltk.translate.gleu_score as gleu
from nlgeval import compute_metrics
from nltk.translate.bleu_score import sentence_bleu

def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def score_gleu(reference, hypothesis):
    score = 0
    for ref, hyp in zip(reference, hypothesis):
        score += gleu.sentence_gleu([ref.split()], hyp.split())
    return float(score) / len(reference)

def compute(preds, gloden):
    t = open(gloden, 'r', encoding='utf8')
    p = open(preds, 'r', encoding='utf8')
    tline = t.readlines()
    pline = p.readlines()
    gleu_result = score_gleu(tline, pline)
    print('GLEU : ', gleu_result)

    metrics_dict = compute_metrics(hypothesis=preds,
                                   references=[gloden], no_skipthoughts=True, no_glove=True)

    return metrics_dict