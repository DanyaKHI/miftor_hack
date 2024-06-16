import pandas as pd
from razdel import tokenize
from pymystem3 import Mystem
import numpy as np


def lemmatize_sample(text):
    m = Mystem()
    tokens = [j.text.lower() for j in tokenize(text)]
    return "".join(m.lemmatize(' '.join(tokens))).strip()


def lemmatize(X):
    return np.array([lemmatize_sample(sample) for sample in X])

