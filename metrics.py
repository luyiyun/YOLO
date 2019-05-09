import numpy as np
import torch


def voc_ap(rec, prec, use_07=False):
    if use_07:
        ap = 0.

