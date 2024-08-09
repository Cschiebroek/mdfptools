import pytest
from utils.data_preprocessing import preprocess_data
import pandas as pd

def test_preprocess_data():
    # Create a mock DataFrame
    data = {
        'molregno': [1, 2, 3],
        'vp_log10_pa': [0.5, 1.0, 1.5],
        'rdkit': [[0, 1], [1, 0], [0, 0]],
        'maccs': [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
    }
    df = pd.DataFrame(data)
    
    train_molregnos, val_molregnos, train_y, val_y, df_train, df_val = preprocess_data(df, 42)
    
    assert not df_train.empty
    assert not df_val.empty
    assert len(train_molregnos) > 0
    assert len(val_molregnos) > 0
