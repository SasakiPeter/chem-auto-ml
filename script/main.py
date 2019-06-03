import os
import pandas as pd
from load_data import load_train_data, load_test_data
from lgb_hyperopt import lgb_opt
from lgb_train_test import lgb_train, lgb_test

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
logger = getLogger(__name__)

DIR = 'result_tmp/'
if not os.path.isdir(DIR):
    os.makedirs(DIR)

def main():
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

    logger.info('start')

    df_train = load_train_data()
    x_train = df_train.loc[:, '0':'2047']
    y_train = df_train['Active_Nonactive'].values
    best_params = lgb_opt(x_train, y_train)
    cutoff, use_cols = lgb_train(x_train, y_train, best_params)

    df_test = load_test_data().sort_values('Name')
    x_test = df_test[use_cols]
    df_all = pd.concat([df_train, df_test], axis=0,
                       sort=False).sort_values('Name')
    x_all = df_all[use_cols]
    lgb_test(x_test, x_all, cutoff)


if __name__ == "__main__":
    main()
