import pytest
from utils.data_preprocessing import preprocess_data,prepare_data
import pandas as pd

def test_preprocess_data():
    conn = 'dbname=cs_mdfps user=cschiebroek host=lebanon'
    df = prepare_data(conn)
    train_molregnos, val_molregnos, train_y, val_y, df_train, df_val = preprocess_data(df, i)
    
    assert not df_train.empty
    assert not df_val.empty
    assert len(train_molregnos) > 0
    assert len(val_molregnos) > 0
