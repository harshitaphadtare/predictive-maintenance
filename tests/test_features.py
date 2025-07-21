import pytest
import pandas as pd

from features import (
    add_rolling_stats,
    add_lag_features,
    scale_features,
    add_delta_features
)

# Sample test DataFrame
@pytest.fixture
def sample_df():
    data = {
        'unit_number': [1, 1, 1, 2, 2, 2],
        'time_in_cycles': [1, 2, 3, 1, 2, 3],
        'sensor_1': [10, 12, 14, 20, 22, 24],
        'sensor_2': [100, 102, 104, 200, 202, 204]
    }
    return pd.DataFrame(data)

def test_add_rolling_stats(sample_df):
    df = add_rolling_stats(sample_df.copy(), sensor_cols=['sensor_1'], window=2)

    # Check columns exist
    assert 'sensor_1_roll_mean' in df.columns
    assert 'sensor_1_roll_std' in df.columns

    # Check no rolling mean for the first row of each unit
    assert pd.isna(df.loc[0, 'sensor_1_roll_mean'])
    assert pd.isna(df.loc[3, 'sensor_1_roll_mean'])

def test_add_lag_features(sample_df):
    df = add_lag_features(sample_df.copy(), sensor_cols=['sensor_2'], lags=[1])

    # Check column exists
    assert 'sensor_2_lag_1' in df.columns

    # Check lag value for first entry per unit should be NaN
    assert pd.isna(df.loc[0, 'sensor_2_lag_1'])
    assert pd.isna(df.loc[3, 'sensor_2_lag_1'])

    # Check that the lagged value is correct
    assert df.loc[1, 'sensor_2_lag_1'] == sample_df.loc[0, 'sensor_2']

def test_scale_features(sample_df):
    feature_cols = ['sensor_1', 'sensor_2']
    df_scaled = scale_features(sample_df.copy(), feature_cols)

    # Values should be between 0 and 1
    assert df_scaled[feature_cols].max().max() <= 1.0
    assert df_scaled[feature_cols].min().min() >= 0.0

def test_add_delta_features(sample_df):
    df = add_delta_features(sample_df.copy(), feature_cols=['sensor_1'])

    # Check column exists
    assert 'sensor_1_delta' in df.columns

    # First row of each unit's delta should be NaN
    assert pd.isna(df.loc[0, 'sensor_1_delta'])
    assert pd.isna(df.loc[3, 'sensor_1_delta'])

    # Check delta value
    expected_delta = sample_df.loc[1, 'sensor_1'] - sample_df.loc[0, 'sensor_1']
    assert df.loc[1, 'sensor_1_delta'] == expected_delta
