#!/usr/bin/python3

from pathlib import Path
import os
from typing import Optional, List, Union, Generator
import pyarrow.parquet as pq
import pyarrow as pa
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class AdamOptimizer:
    def __init__(self, params, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params.keys():
            # Moving averages of gradient and squared gradient
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[f"d{k}"]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[f"d{k}"] ** 2)

            # Bias correction
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            # Parameter update
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
