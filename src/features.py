from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

#rolling statistics function
def add_rolling_stats(df,sensor_cols,window=3):
    for col in sensor_cols:
        #calculating rolling mean
        roll_mean = df.groupby('unit_number')[col].rolling(window).mean()
        #calculating rolling std
        roll_std = df.groupby('unit_number')[col].rolling(window).std()

        #adding rolling mean and std to df
        df[f'{col}_roll_mean'] = roll_mean.reset_index(0,drop=True)
        df[f'{col}_roll_std'] = roll_std.reset_index(0,drop=True)

    return df

#lag feature function
def add_lag_features(df,sensor_cols,lags=[1,2]):
    for col in sensor_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby('unit_number')[col].shift(lag)

    return df


#Normalizing Features
def scale_features(df,feature_cols):
    scaler = MinMaxScaler()
    df.loc[:, feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

#delta feature
def add_delta_features(df,feature_cols):
    for col in feature_cols:
        df[f'{col}_delta'] = df.groupby('unit_number')[col].diff()
    return df

#dropping highly correlated features
def drop_correlated_features(df,feature_columns,threshold=0.95):
   
    corr_matrix = df[feature_columns].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    cleaned_df = df.drop(to_drop,axis=1)
    
    return cleaned_df,to_drop

