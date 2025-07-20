from sklearn.preprocessing import MinMaxScaler

#rolling statistics function
def add_rolling_stats(df,sensor_cols,window=5):
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
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

#delta feature
def add_delta_features(df,feature_cols):
    for col in feature_cols:
        df[f'{col}_delta'] = df.groupby('unit_number')[col].diff()
    return df