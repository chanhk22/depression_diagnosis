# preprocessing/utils_io.py
import pandas as pd
import os

def read_table_smart(path):
    """
    CSV가 쉼표/세미콜론/탭 등 섞여 있어도 자동 파싱.
    """
    # 1) 먼저 , 으로 시도
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df, ','
    except Exception:
        pass
    # 2) ; 시도
    try:
        df = pd.read_csv(path, sep=';')
        if df.shape[1] > 1:
            return df, ';'
    except Exception:
        pass
    # 3) \t 시도
    try:
        df = pd.read_csv(path, sep='\t')
        if df.shape[1] > 1:
            return df, '\t'
    except Exception:
        pass
    # 마지막으로 기본
    return pd.read_csv(path, engine='python'), ','
