import random
import os
os.environ ["CUDA_VISIBLE_DEVICES"] = "1"
import timeit

import pandas as pd

from model import AttentionDTI
from dataset import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
from pytorchtools import EarlyStopping
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc


def show_result(DATASET, label, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The {} model's results:".format(label))
    with open("./{}/results.txt".format(DATASET), 'a') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]


def test_process(model, pbar, LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            compounds, proteins, labels = data
            compounds = compounds.cuda()
            proteins = proteins.cuda()
            labels = labels.cuda()

            predicted_scores = model(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())

    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC


def test_model(dataset_load, save_path, DATASET, LOSS, dataset="Train", label="best", save=True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_process(model, test_pbar, LOSS)

    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET, dataset, label), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(label, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


if __name__ == "__main__":
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    hp = hyperparameter()

    DATASET = "BindingDB"  # 可以选择其他数据集，如DrugBank或Davis
    print("Train in " + DATASET)

    # 手动输入训练集和测试集路径
    dir_input_train = "/root/bindingdb/train_val_label.csv"
    dir_input_test = "/root/bindingdb/test_label.csv"

    print(f"加载训练数据：{dir_input_train}")
    train_data_list = pd.read_csv(dir_input_train)
    print(f"加载测试数据：{dir_input_test}")
    test_data_list = pd.read_csv(dir_input_test)

    print("训练集和测试集数据加载成功")

    # 加载训练和测试数据集
    train_dataset = CustomDataSet(train_data_list)
    test_dataset = CustomDataSet(test_data_list)

    # 使用DataLoader加载数据集
    train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                   collate_fn=collate_fn)

    # 创建模型
    model = AttentionDTI(hp).cuda()
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    optimizer = optim.AdamW(
        [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}],
        lr=hp.Learning_rate)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,
                                            cycle_momentum=False, step_size_up=len(train_dataset_load) // hp.Batch_size)
    Loss = nn.CrossEntropyLoss()

    save_path = "./" + DATASET
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_results = save_path + 'The_results_of_whole_dataset.txt'
    with open(file_results, 'w') as f:
        hp_attr = '\n'.join(['%s:%s' % item for item in hp.__dict__.items()])
        f.write(hp_attr + '\n')

    early_stopping = EarlyStopping(savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

    # 开始训练
    print('开始训练...')
    start = timeit.default_timer()

    # 用于存储各项指标
    Accuracy_List_stable = []
    Precision_List_stable = []
    Recall_List_stable = []
    AUC_List_stable = []
    AUPR_List_stable = []

    for epoch in range(1, hp.Epoch + 1):
        train_pbar = tqdm(
            enumerate(
                BackgroundGenerator(train_dataset_load)),
            total=len(train_dataset_load))

        train_losses_in_epoch = []
        model.train()
        for train_i, train_data in train_pbar:
            trian_compounds, trian_proteins, trian_labels = train_data
            trian_compounds = trian_compounds.cuda()
            trian_proteins = trian_proteins.cuda()
            trian_labels = trian_labels.cuda()

            optimizer.zero_grad()

            predicted_interaction = model(trian_compounds, trian_proteins)
            train_loss = Loss(predicted_interaction, trian_labels)
            train_losses_in_epoch.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss_a_epoch = np.average(train_losses_in_epoch)
        print(f'Epoch {epoch} Training loss: {train_loss_a_epoch:.4f}')

        # 每隔一定epoch进行测试并记录结果
        if epoch % hp.Eval_freq == 0:
            print(f"正在评估模型在测试集上的表现...")
            results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
                test_model(test_dataset_load, save_path, DATASET, Loss, dataset="Test", label=str(epoch))

            # 存储每次测试的指标
            Accuracy_List_stable.append(Accuracy_test)
            Precision_List_stable.append(Precision_test)
            Recall_List_stable.append(Recall_test)
            AUC_List_stable.append(AUC_test)
            AUPR_List_stable.append(PRC_test)

            # 调用show_result函数来显示当前结果
            show_result(DATASET, "stable",
                        Accuracy_List_stable, Precision_List_stable, Recall_List_stable,
                        AUC_List_stable, AUPR_List_stable)

    print(f'训练完成，耗时：{timeit.default_timer() - start:.2f}秒。')



