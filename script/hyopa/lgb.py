import os
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials
import lightgbm as lgb

from tqdm import tqdm
import pickle
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = '../result_tmp/'
if not os.path.isdir(DIR):
    os.makedirs(DIR)


def lgb_opt_params(x_train, y_train):

    log_fmt = Formatter(
        '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train_lgb_clf_hyperopt.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('hyper optimization start')

    use_cols = x_train.columns.values

    def objective(params):
        dtrain = lgb.Dataset(
            x_train,
            label=y_train,
            feature_name=use_cols.tolist()
        )
        cv_results = lgb.cv(
            params,
            dtrain,
            nfold=5,
            num_boost_round=20000,
            early_stopping_rounds=150,
            metrics='auc',
            seed=0
        )

        loss = 1 - max(cv_results['auc-mean'])
        return loss

    space = {
        'min_child_weight': hp.uniform('min_child_weight', 0, 1000),
        'subsample': hp.uniform('subsample', 0, 1),
        'subsample_freq': hp.choice('subsample_freq', np.arange(0, 2, dtype=int)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
        'learning_rate': hp.uniform('learning_rate', 1e-8, 0.1),
        'max_depth': hp.choice('max_depth', np.arange(1, 100, dtype=int)),
        'min_split_gain': hp.uniform('min_split_gain', 0, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'max_bin': 255,
        'num_leaves': hp.choice('num_leaves', np.arange(2, 100, dtype=int)),
        'objective': 'xentropy',
        'scale_pos_weight': hp.uniform('scale_pos_weight', 0, 1000),
        'verbose': -1,
        'boosting_type': 'gbdt',
        'metric': 'auc',
        # 'device': 'gpu',
        # 'gpu_device_id': 1,
        'nthread': -1,
        'seed': 0,
    }

    trials = Trials()

    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                trials=trials, max_evals=150)

    logger.info('best params {}'.format(best))
    logger.info('hyper optimization end')
    return best


if __name__ == "__main__":
    df_train = load_train_data()
    x_train = df_train.loc[:, '0':'2047']
    y_train = df_train['Active_Nonactive'].values
    lgb_opt_params(x_train, y_train)
