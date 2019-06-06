import os
import pandas as pd
from load_data import load_train_data, load_test_data
from hyopa import lgb_opt_params
from lgb_train_test import opt_iter, create_models, predict_test

from logging import StreamHandler, DEBUG, INFO, Formatter, FileHandler, getLogger
logger = getLogger(__name__)

DIR = 'result_tmp/'
if not os.path.isdir(DIR):
    os.makedirs(DIR)


def main():
    log_fmt = Formatter(
        '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train_lgb_clf_hyperopt.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    logger.info("start exploring best params")

    logger.info("start exploring best params without iteration")
    df_train = load_train_data()
    x_train = df_train.loc[:, '0':'2047']
    y_train = df_train['Active_Nonactive'].values
    best_params = lgb_opt_params(x_train, y_train)
    logger.info("end exploring best params without iteration")

    logger.info("start optimizing iteration")
    best_iter = opt_iter(x_train, y_train, best_params)
    logger.info("end optimizing iteration")

    logger.info("end exploring best params")

    logger.info("start best params train")
    best_model_No, cutoff = create_models(
        x_train, y_train, best_params, best_iter)
    logger.info("end best params train")

    logger.info("start predict unknown data(test data)")
    df_test = load_test_data().sort_values('Name')
    use_cols = x_train.columns.values
    # x_test = df_test[use_cols]
    df_all = pd.concat([df_train, df_test], axis=0,
                       sort=False).sort_values('Name')
    x_all = df_all[use_cols]
    predict_test(x_all, best_model_No, cutoff)
    logger.info("end predict unknown data(test data)")

    logger.info("end")


if __name__ == "__main__":
    main()
