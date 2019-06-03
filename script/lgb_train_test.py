import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import lightgbm as lgb

from tqdm import tqdm
import pickle
import gc
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
if not os.path.isdir(DIR):
    os.makedirs(DIR)

SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'
ALL_SAMPLE_SUBMIT_FILE = '../input/sample_submission_all.csv'


def to_bin(pred, cutoff):
    return [1 if p >= cutoff else 0 for p in pred]


def lgb_train(x_train, y_train, params):

    log_fmt = Formatter(
        '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train_lgb_clf.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')
    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
    logger.info('data preparation end {}'.format(x_train.shape))

    all_params = {
        'min_child_weight': [params['min_child_weight']],
        'subsample': [params['subsample']],
        'subsample_freq': [params['subsample_freq']],
        'seed': [0],
        'colsample_bytree': [params['colsample_bytree']],
        'learning_rate': [params['learning_rate']],
        'max_depth': [params['max_depth']],
        'min_split_gain': [params['min_split_gain']],
        'reg_alpha': [params['reg_alpha']],
        'reg_lambda': [params['reg_lambda']],
        'max_bin': [255],
        'num_leaves': [params['num_leaves']],
        'objective': ['xentropy'],
        'scale_pos_weight': [params['scale_pos_weight']],
        'verbose': [-1],
        'boosting_type': ['gbdt'],
        'metric': ['auc'],
        'nthreads': [-1],
        # 'device': [gpu],
        # 'gpu_device_id': [0],
        # 'skip_drop': [0.7],
    }

    max_score = 0
    min_params = None
    best_iter = 1

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))
        list_auc_score = []
        list_best_iterations = []

        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            dtrain = lgb.Dataset(
                trn_x.values,
                label=trn_y,
                feature_name=use_cols.tolist()
            )

            dvalid = lgb.Dataset(
                val_x.values,
                label=val_y,
                feature_name=use_cols.tolist()
            )
            del trn_x
            gc.collect()

            clf = lgb.train(
                params,
                dtrain,
                20000,
                valid_sets=[dtrain, dvalid],
                early_stopping_rounds=150,
                # feval='auc',
                verbose_eval=1
            )

            pred = clf.predict(val_x)
            sc_auc = roc_auc_score(val_y, pred)

            list_auc_score.append(sc_auc)
            list_best_iterations.append(clf.best_iteration)

            logger.debug('auc: {}, iter: {}'.format(
                sc_auc, clf.best_iteration))
        sc_auc = np.mean(list_auc_score)
        it = int(np.mean(list_best_iterations))

        if max_score < sc_auc:
            max_score = sc_auc
            min_params = params
            best_iter = it

        logger.info('current max score(auc): {}, best iter: {}, params: {}'.format(
            max_score, best_iter, min_params))

    logger.info('minimum params: {}'.format(min_params))
    logger.info('maximum auc: {}'.format(max_score))

    logger.info('min params train start')

    dtrain = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=use_cols.tolist()
    )

    clf = lgb.train(
        min_params,
        dtrain,
        best_iter,
    )

    logger.info('train end')

    pred_train = clf.predict(x_train)
    best_bac = 0
    best_cutoff = 0

    for cutoff in np.arange(0, 1, 0.01):
        pred_train_bin = to_bin(pred_train, cutoff)
        bac = balanced_accuracy_score(y_train, pred_train_bin)
        if bac > best_bac:
            best_bac = bac
            best_cutoff = cutoff

    logger.info('best cutoff: {}, best bac {}'.format(best_cutoff, best_bac))

    with open(DIR + 'model_lgb_clf.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)

    return best_cutoff, use_cols


def lgb_test(x_test, x_all, best_cutoff):
    with open(DIR + 'model_lgb_clf.pkl', 'rb') as f:
        clf = pickle.load(f)

    logger.info('predict test data start')
    pred_test = clf.predict(x_test)
    logger.info('predict test data end {}'.format(pred_test.shape))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('Name')
    df_submit['Prediction'] = to_bin(pred_test, best_cutoff)

    df_submit.to_csv(DIR + 'submit_clf.csv', index=False)

    logger.info('predict all data start')
    pred_all = clf.predict(x_all)
    logger.info('predict all data end {}'.format(pred_all.shape))

    df_submit_all = pd.read_csv(ALL_SAMPLE_SUBMIT_FILE).sort_values('Name')
    df_submit_all['Prediction'] = to_bin(pred_all, best_cutoff)
    df_submit_all.to_csv(DIR + 'submit_clf_all.csv', index=False)

    logger.info('end')
