import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    """
    将SMILES字符串转化为整数序列。如果字符不在字典中，使用默认值0。
    """
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        # 如果字符不在字典中，返回默认值0
        X[i] = smi_ch_ind.get(ch, 0)
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    """
    将蛋白质序列转化为整数序列。如果字符不在字典中，使用默认值0。
    """
    X = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        # 如果字符不在字典中，返回默认值0
        X[i] = smi_ch_ind.get(ch, 0)
    return X


class CustomDataSet(Dataset):
    def __init__(self, data_frame):
        """
        Args:
            data_frame (pd.DataFrame): 包含所有样本数据的DataFrame，需包含以下列：
                                       'compound', 'protein', 'label'
        """
        self.data_frame = data_frame

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        return row['Drug'], row['Target'], row['Label']

    def __len__(self):
        return len(self.data_frame)

def collate_fn(batch_data):
    N = len(batch_data)
    compound_max = 100
    protein_max = 1000
    compound_new = torch.zeros((N, compound_max),dtype=torch.long)
    protein_new = torch.zeros((N, protein_max),dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i, (compoundstr, proteinstr, label) in enumerate(batch_data):
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        labels_new[i] = int(label)

    return compound_new, protein_new, labels_new