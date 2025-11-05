import numpy as np
import pandas as pd
import argparse
import os

#os.chdir("D:/Battery/papers/RUL-Mamba - Original/FuseDR-Net_TJU")

class DF():
    # ... (This class remains as before) ...
    def __init__(self, args):
        self.args = args

    def handle_outliers_by_interpolation(self, df: pd.DataFrame, columns_to_clean: list) -> pd.DataFrame:
        df_out = df.copy()
        for col in columns_to_clean:
            if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
                mean = df_out[col].mean()
                std = df_out[col].std()
                upper_bound = mean + 3 * std
                lower_bound = mean - 3 * std
                outliers = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
                df_out.loc[outliers, col] = np.nan
        df_out[columns_to_clean] = df_out[columns_to_clean].interpolate(method='linear', limit_direction='both', axis=0)
        return df_out

    def read_one_csv(self, file_name):
        df = pd.read_csv(file_name)
        df['Cycle'] = np.arange(1, len(df) + 1)
        return df

# --- New method for decomposition based on your custom algorithm ---

def custom_monotonic_decomposition(capacity_signal):
    """
    This function implements your custom rule-based algorithm to create a strictly decreasing trend.
    """
    n_points = len(capacity_signal)
    if n_points == 0:
        return np.array([]), np.array([])

    monotonic_trend = np.zeros(n_points)
    
    # Step 1: Initialization
    monotonic_trend[0] = capacity_signal[0]
    
    # Step 2: Iterate and apply the condition
    for t in range(1, n_points):
        if capacity_signal[t] < monotonic_trend[t-1]:
            monotonic_trend[t] = capacity_signal[t]
        else:
            monotonic_trend[t] = monotonic_trend[t-1]
            
    # Step 3: Calculate the residual
    residual_noise = capacity_signal - monotonic_trend
    
    return monotonic_trend, residual_noise


def BatteryDataRead(args):
    """
    The main function that now uses your custom decomposition algorithm.
    """
    root = 'data/TJU data/Dataset_3_NCM_NCA_battery'
    files = os.listdir(root)
    files = [f for f in os.listdir(root) if f.lower().endswith(".csv")]

    Battery_list = {}
    
    health_indicator_columns = [
        'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy'
    ]

    print("Starting CUSTOM preprocessing for NASA dataset...")
    
    for i in range(1,4):
        for f in files:
            if 'CY25-05_1' in f and '#'+str(i) in f:
                path = os.path.join(root, f)
                df_i = DF(args)

                # 1. Read raw data
                df = df_i.read_one_csv(path)
                df = df.rename(columns={'capacity': 'Capacity'})
                Battery_name, _ = os.path.splitext(f)
                df['BatteryName'] = f'CY25_{i}'

                df = df_i.handle_outliers_by_interpolation(df, health_indicator_columns)
                
                capacity_signal = df['Capacity'].values
                
                # *** Key change: Calling your custom decomposition function ***
                monotonic_trend, residual_noise = custom_monotonic_decomposition(capacity_signal)
                
                df['monotonic_trend'] = monotonic_trend
                df['residual_noise'] = residual_noise

                Battery_list[f'CY25_{i}'] = df
                print(f"Finished custom processing for {Battery_name}")
                        
    return Battery_list

# The MultiVariateBatteryDataProcess function remains unchanged
def MultiVariateBatteryDataProcess(BatteryData, test_name, start_point, args):
    """
    *** Finalized function with unified normalization logic ***
    Now 'residual_noise' is also normalized to the [0, 1] range using Min-Max, just like other features.
    """
    dict_without_test = {key: value for key, value in BatteryData.items() if key != test_name}
    df_train = pd.concat(dict_without_test.values())
    
    feature_columns = ['BatteryName', 'Cycle',
                       'voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy',
                       'current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy',
                       'monotonic_trend', 'residual_noise', 'Capacity']
    df_train = df_train.filter(items=feature_columns)
    
    df_train['target'] = df_train['Capacity'] / args.Rated_Capacity
    
    # --- Unified normalization ---
    numeric_cols = df_train.select_dtypes(include=np.number).columns.tolist()
    if 'Cycle' in numeric_cols:
        numeric_cols.remove('Cycle') 
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
            
    scalers = {}
    # *** Key change: The if/else condition for residual_noise has been removed ***
    # Now all features are uniformly normalized with Min-Max.
    for col in numeric_cols:
        min_val = df_train[col].min()
        max_val = df_train[col].max()
        scalers[col] = ('minmax', min_val, max_val)
        if max_val > min_val:
            df_train[col] = (df_train[col] - min_val) / (max_val - min_val)
        else:
            df_train[col] = 0
            
    df_train['time_idx'] = df_train['Cycle'].map(lambda x: int(x - 1))
    df_train['group_id'] = df_train['BatteryName'].map({'CY25_1':0, 'CY25_2':1, 'CY25_3':2})
    df_train = df_train.drop(['BatteryName', 'Capacity'], axis=1)
    df_train['idx'] = range(len(df_train))
    df_train.set_index('idx', inplace=True)

    #df_train.to_csv('df_train_costume2.csv')

    # --- Processing test data ---
    df_test = BatteryData[test_name]
    df_test = df_test.filter(items=feature_columns)
    df_test['target'] = df_test['Capacity'] / args.Rated_Capacity

    # The logic here is also simplified because we only have one type of scaler
    for col in numeric_cols:
        scaler_type, min_val, max_val = scalers[col]
        if max_val > min_val:
            df_test[col] = (df_test[col] - min_val) / (max_val - min_val)
        else:
            df_test[col] = 0
            
    df_test['time_idx'] = df_test['Cycle'].map(lambda x: int(x-1))
    df_test['group_id'] = df_test['BatteryName'].map({'CY25_1':0, 'CY25_2':1, 'CY25_3':2})
    df_test = df_test.drop(['BatteryName', 'Capacity'], axis=1)
    df_test['idx'] = range(len(df_test))
    df_test.set_index('idx', inplace=True)
    
    df_all = df_test
    
    feature_columns_for_test = [col for col in df_train.columns if col not in ['target']]
    df_test = df_all.loc[df_all['Cycle'] >= start_point - args.seq_len, feature_columns_for_test + ['target']]
    df_test['idx'] = range(len(df_test))
    df_test.set_index('idx', inplace=True)
    
    #df_test.to_csv('df_test_costume2.csv')

    return df_train, df_test, df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set your parameters for NASA here
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--Rated_Capacity', type=float, default=2.5)
    parser.add_argument('--test_name', type=str, default='CY25_1')
    parser.add_argument('--start_point_list', type=list, default=[200,300,400])
    args = parser.parse_args()
    
    BatteryData = BatteryDataRead(args)
    # Change the output filename for distinction
    np.save('data/TJU data/TJU_dataset.npy', np.array([BatteryData], dtype=object))
    print("Saved causally processed NASA data to 'TJU_dataset.npy'")


    #BatteryData = np.load('data/TJU data/TJU_dataset.npy', allow_pickle=True)
    #BatteryData = BatteryData.item()
    #_,_,df_all = MultiVariateBatteryDataProcess(BatteryData,args.test_name,args.start_point_list[0],args)