import os
import pandas as pd
from config import BASE_RAW_PATH, BASE_PROCESSED_PATH


def load_data(dataset_id="FD001", dataset_type="train"):

    '''
    Loads any dataset (but default is FD001) from the raw data directory.
    Returns a pandas dataframe with the following columns:
    - unit_number: the unit number
    - time_in_cycles: the number of cycles the unit has run
    - op_setting_1 to op_setting_3: three operating settings
    - sensor_1 to sensor_21: 21 sensors
    '''
    
    col_names = ['unit_number','time_in_cycles'] + \
                [f'op_setting_{i}' for i in range(1,4)] + \
                [f'sensor_{i}' for i in range(1,22)] 

    file_path = os.path.join(BASE_RAW_PATH, dataset_id, f'{dataset_type}_{dataset_id}.txt')
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    
    #drop empty columns due to irregular spacing
    df.dropna(axis=1, how='all', inplace=True)
    #rename columns
    df.columns = col_names

    #convert time_in_cycles and unit_number to int
    df['unit_number'] = df['unit_number'].astype(int)
    df['time_in_cycles'] = df['time_in_cycles'].astype(int)

    return df



def generate_labels(dataset_id="FD001", dataset_type="train",failure_threshold=30):

    '''
        Generates RUL labels and binary near-failure 
        classification labels
        train: for every engine we calculate rul
        test: engines do not run to failure, so we need to 
        use the RUL_FD001.txt file to assign the actual remaining 
        life for each test unit at its last recorded cycle.
    '''
    
    if dataset_type == "train":
        # Use load_data function for consistency
        df = load_data(dataset_id=dataset_id, dataset_type=dataset_type)

        #regression target
        df['RUL'] = df.groupby('unit_number')['time_in_cycles'].transform('max') - df['time_in_cycles']

        #binary classification target
        df['label'] = df['RUL'].apply(lambda x: 1 if x <= failure_threshold else 0)

        #save data to processed 
        output_path = os.path.join(BASE_PROCESSED_PATH, dataset_id, f'train_{dataset_id}.csv')
        df.to_csv(output_path,index=False)

        return df

    else: 
        # Use load_data function for consistency
        df_test = load_data(dataset_id=dataset_id, dataset_type=dataset_type)
        
        rul_path = os.path.join(BASE_RAW_PATH, dataset_id, f'RUL_{dataset_id}.txt')
        
        #Get last cycle for each unit
        last_cycles = df_test.groupby('unit_number')['time_in_cycles'].max().reset_index() 

        #loading RUL values
        rul = pd.read_csv(rul_path,header=None,names=['RUL'])

        #combine RUL with last cycles
        last_cycles['RUL'] = rul['RUL']

        #merge RUL into test data
        df_test = df_test.merge(last_cycles,on=['unit_number','time_in_cycles'],how='left')

        # Add binary classification label
        df_test['label'] = df_test['RUL'].apply(lambda x: 1 if x <= failure_threshold else 0)

        output_path = os.path.join(BASE_PROCESSED_PATH, dataset_id, f'test_{dataset_id}.csv')
        df_test.to_csv(output_path,index=False)

        return df_test

