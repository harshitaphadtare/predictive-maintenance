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

    file_path = BASE_RAW_PATH + f'/{dataset_id}/{dataset_type}_{dataset_id}.txt'
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    
    #drop empty columns due to irregular spacing
    df.dropna(axis=1, how='all', inplace=True)
    #rename columns
    df.columns = col_names

    #convert time_in_cycles and unit_number to int
    df['unit_number'] = df['unit_number'].astype(int)
    df['time_in_cycles'] = df['time_in_cycles'].astype(int)

    return df



def generate_labels(dataset_id="FD001", dataset_type="train"):

    '''
        Generates RUL labels for datasets. Returns Processed 
        DataFrame with RUL column.

        train: for every engine we calculate rul
        test: engines do not run to failure, so we need to 
        use the RUL_FD001.txt file to assign the actual remaining 
        life for each test unit at its last recorded cycle.
    '''
    
    if dataset_type == "train":

        train_path = BASE_RAW_PATH + f'/{dataset_id}/train_{dataset_id}.txt'

        df = pd.read_csv(train_path,sep=' ',header=None)
        df.drop(columns=[26,27],inplace=True) #Remove trailing empty columns
        
        df.columns = ['unit_number','time_in_cycles'] + \
                [f'op_setting_{i}' for i in range(1,4)] + \
                [f'sensor_{i}' for i in range(1,22)] 

        df['RUL'] = df.groupby('unit_number')['time_in_cycles'].transform('max') - df['time_in_cycles']

        #save data to processed 
        output_path = BASE_PROCESSED_PATH + f'/{dataset_id}/train_{dataset_id}.csv'
        df.to_csv(output_path,index=False)

        return df

    else: 

        test_path = BASE_RAW_PATH + f'/{dataset_id}/test_{dataset_id}.txt'
        rul_path = BASE_RAW_PATH + f'/{dataset_id}/RUL_{dataset_id}.txt'

        df_test = pd.read_csv(test_path,sep=' ',header=None)
        df_test.drop(columns=[26,27],inplace=True) 
        df_test.columns = ['unit_number','time_in_cycles'] + \
                [f'op_setting_{i}' for i in range(1,4)] + \
                [f'sensor_{i}' for i in range(1,22)] 
        
        #Get last cycle for each unit
        last_cycles = df_test.groupby('unit_number')['time_in_cycles'].max().reset_index() 

        #loading RUL values
        rul = pd.read_csv(rul_path,header=None,names=['RUL'])

        #combine RUL with last cycles
        last_cycles['RUL'] = rul['RUL']

        #merge RUL into test data
        df_test = df_test.merge(last_cycles,on=['unit_number','time_in_cycles'],how='left')

        output_path = BASE_PROCESSED_PATH + f'/{dataset_id}/test_{dataset_id}.csv'
        df_test.to_csv(output_path,index=False)

        return df_test




    
