import pandas as pd
from src.config import BASE_DATASET_PATH

def load_data(dataset_id="FD001", dataset_type="train"):

    '''
    Loads the any dataset (but default is FD001) from the raw data directory.
    Returns a pandas dataframe with the following columns:
    - unit_number: the unit number
    - time_in_cycles: the number of cycles the unit has run
    - op_setting_1 to op_setting_3: three operating settings
    - sensor_1 to sensor_21: 21 sensors
    '''
    
    col_names = ['unit_number','time_in_cycles'] + \
                [f'op_setting_{i}' for i in range(1,4)] + \
                [f'sensor_{i}' for i in range(1,22)] 

    file_path = BASE_DATASET_PATH + f'/{dataset_id}/{dataset_type}_{dataset_id}.txt'
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    
    #drop empty columns due to irregular spacing
    df.dropna(axis=1, how='all', inplace=True)
    #rename columns
    df.columns = col_names

    #convert time_in_cycles and unit_number to int
    df['unit_number'] = df['unit_number'].astype(int)
    df['time_in_cycles'] = df['time_in_cycles'].astype(int)

    return df
    
