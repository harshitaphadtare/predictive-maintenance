import pandas as pd
from data_prep import load_data, generate_labels

#testing data loading
def test_load_data_structure():

    df = load_data('FD001','train')

    #Basic shape and type testing
    assert isinstance(df,pd.DataFrame)
    assert 'unit_number' in df.columns
    assert 'time_in_cycles' in df.columns
    assert df.shape[1] == 26

    #check types
    assert df['unit_number'].dtype == 'int'
    assert df['time_in_cycles'].dtype == 'int'


#testing generation of labels for train data
def test_generate_labels_train():

    df = generate_labels('FD001','train')

    assert 'RUL' in df.columns
    assert df['RUL'].min() == 0
    assert df.shape[0] > 0


#testing generation of labels for test data
def test_generate_labels_test():

    df = generate_labels('FD001','test')

    assert 'RUL' in df.columns
    assert df.shape[0] > 0

    #checking if last cycles have non_null RULs
    non_null_rul = df[df['RUL'].notnull()]
    assert (non_null_rul.groupby('unit_number')['time_in_cycles'].idxmax() == non_null_rul.index).all()
