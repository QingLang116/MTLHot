# 本模块用于导入处理后的基因组变异数据以及炎性表型数据
import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_feature_data(filename):
    data = pd.read_csv(filename, sep=',', index_col=0)
    features = data.columns.values
    sample_names = data.index.values
    data = data.to_numpy()
    return data, features, sample_names


def load_phenotype_data(filename):
    pheno_score = pd.read_csv(filename, sep=',', index_col=0)
    pheno_labels = pheno_score.columns.values
    sample_names = pheno_score.index.values
    return pheno_score, pheno_labels, sample_names


def load_final_data(X, Y_labels, shuffle=True):
    np.random.seed(0)
    # -----------对全部的特征进行归一化处理---------------------#
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_X = min_max_scaler.fit_transform(X)

    if shuffle is True:
        # ------------ 打乱样本顺序------------------------------ #
        shuffle_idx = np.random.permutation(range(X.shape[0]))
        X_shuffle = normalized_X[shuffle_idx]
        Y_labels_shuffle = Y_labels.iloc[shuffle_idx, :]

        # ------------ 将dataFrame格式转化为dict----------------- #
        phenotypes = Y_labels_shuffle.columns.values  # 获得表型名称
        Y = {}
        for phen in phenotypes:
            Y[phen] = Y_labels_shuffle[phen].astype(float).values

        return X_shuffle, Y, shuffle_idx
    else:
        # ------------ 将dataFrame格式转化为dict----------------- #
        phenotypes = Y_labels.columns.values  # 获得表型名称
        Y = {}
        for phen in phenotypes:
            Y[phen] = Y_labels[phen].astype(float).values
        return normalized_X, Y


def normalized_val_data(val_feature_data, train_feature_data):
    scaled_val_exp_data = np.zeros((val_feature_data.shape[0], val_feature_data.shape[1]))
    if val_feature_data.shape[1] == train_feature_data.shape[1]:
        for i in range(train_feature_data.shape[1]):
            train_values = train_feature_data[:, i]
            val_values = val_feature_data[:, i]
            train_mean = np.mean(train_values)
            train_sd = np.std(train_values)
            val_mean = np.mean(val_values)
            val_sd = np.std(val_values)
            rescaled_value = ((val_values - val_mean) * train_sd / val_sd) + train_mean

            scaled_val_exp_data[:, i] = rescaled_value

        return scaled_val_exp_data
    else:
        print("Different feature numbers!")
        return val_feature_data
