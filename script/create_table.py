# coding: utf-8
import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from mordred import Calculator, descriptors
import mordred

from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

logger = getLogger(__name__)


TRAIN_SOURCE = '../input/pubchem_aid1996_moe_3d_generated_testout.sdf'
TEST_SOURCE = '../input/pubchem_aid1996_moe_3d_generated_test_ansout.sdf'

TRAIN_OUTPUT = '../input/train.csv'
TEST_OUTPUT = '../input/test.csv'

DIR = 'result_tmp/'
if not os.path.isdir(DIR):
    os.makedirs(DIR)

log_fmt = Formatter(
    '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(DIR + 'create_table.py.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)


# sourceはsdf形式
def extract_descriptor(source, output):
    logger.info("start load sdf")
    df = PandasTools.LoadSDF(source)
    # df.Sol_pH74_Mean = df.Sol_pH74_Mean.astype(float)
    logger.info("input shape {}".format(df.shape))
    logger.info("end load sdf")

    logger.info("start calc fps")
    mols = [x for x in df.ROMol]
    fps = [[x for x in AllChem.GetMorganFingerprintAsBitVect(
        mol, 2, 2048)] for mol in mols]
    fps = pd.DataFrame(fps)
    logger.info("end calc fps {}".format(fps.shape))

    logger.info("start calc descs")
    calc = Calculator(descriptors, ignore_3D=False)
    descs = calc.pandas(mols)
    logger.info("end calc descs {}".format(descs.shape))

    logger.info("start preprocessing")
    for i, col_val in tqdm(enumerate(descs.values.tolist())):
        for j, val in enumerate(col_val):
            if isinstance(val, mordred.error.Missing):
                descs.iloc[i, j] = 0
            elif isinstance(val, bool):
                if val == True:
                    descs.iloc[i, j] = 1
                elif val == False:
                    descs.iloc[i, j] = 0
    logger.info("end preprocessing")

    summary = pd.concat([df, descs, fps], axis=1)
    logger.info("summary shape {}".format(summary.shape))
    logger.info("start write out")
    summary.to_csv(output, encoding='utf-8', index=False)
    logger.info("end write out")


def main():
    # extract_descriptor(TRAIN_SOURCE, TRAIN_OUTPUT)
    extract_descriptor(TEST_SOURCE, TEST_OUTPUT)


if __name__ == "__main__":
    main()
