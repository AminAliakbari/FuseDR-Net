import numpy as np
import pandas as pd
import scipy.io
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import skew, entropy, kurtosis
from scipy.integrate import simpson
from pathlib import Path
import logging
import argparse
from typing import Any, List, Union

# --- Configuration ---
COLUMN_ORDER = [
    'voltage mean', 'voltage std', 'voltage skewness', 'voltage kurtosis', 'CC Q',
    'CC charge time', 'voltage slope', 'voltage entropy', 'current mean', 'current std',
    'current skewness', 'current kurtosis', 'CV Q', 'CV charge time',
    'current slope', 'current entropy', 'capacity'
]

def setup_logging(log_level: str, log_file: Path):
    """Configures logging to output to both the console and a file."""
    log_level = log_level.upper()
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

def safe_get(data_struct: Any, path: List[Union[str, int]], default: Any = None) -> Any:
    """Safely retrieves a nested value from a complex structure."""
    current_level = data_struct
    for key in path:
        try:
            current_level = current_level[key]
        except (KeyError, IndexError, TypeError):
            return default
    return current_level

def _perform_interpolation(df: pd.DataFrame, time_step: float, voltage_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Helper function to perform all interpolation steps."""
    try:
        t_uniform = np.arange(df['time(s)'].min(), df['time(s)'].max() + time_step, time_step)
        
        v_mask = df['voltage(v)'] >= voltage_threshold
        if v_mask.sum() < 2:
            return None
        t_cc_end = df[v_mask]['time(s)'].iloc[0]
        t_cv_start = df[v_mask]['time(s)'].iloc[1]

        df_cc = df[df['time(s)'] <= t_cc_end]
        df_cv = df[df['time(s)'] > t_cc_end]
        if len(df_cc) < 2 or len(df_cv) < 2: return None

        df_cv_seeded = pd.concat([df_cc.iloc[-1:], df_cv], ignore_index=True)
        df_cc_seeded = pd.concat([df_cc, df_cv.iloc[0:1]], ignore_index=True)

        interp_voltage_cc = CubicSpline(df_cc['time(s)'], df_cc['voltage(v)'])
        interp_voltage_cv = interp1d(df_cv_seeded['time(s)'], df_cv_seeded['voltage(v)'], kind='previous', fill_value='extrapolate')
        interp_current_cc = interp1d(df_cc_seeded['time(s)'], df_cc_seeded['current(A)'], kind='next', fill_value='extrapolate')
        interp_current_cv = CubicSpline(df_cv['time(s)'], df_cv['current(A)'])

        voltage_interp = np.zeros_like(t_uniform)
        current_interp = np.zeros_like(t_uniform)

        is_voltage_cc = t_uniform <= t_cc_end
        is_voltage_cv = t_uniform > t_cc_end
        is_current_cc = t_uniform <= t_cv_start
        is_current_cv = t_uniform > t_cv_start
        
        voltage_interp[is_voltage_cc] = interp_voltage_cc(t_uniform[is_voltage_cc])
        voltage_interp[is_voltage_cv] = interp_voltage_cv(t_uniform[is_voltage_cv])
        current_interp[is_current_cc] = interp_current_cc(t_uniform[is_current_cc])
        current_interp[is_current_cv] = interp_current_cv(t_uniform[is_current_cv])

        df_interp = pd.DataFrame({'time(s)': t_uniform, 'voltage(v)': voltage_interp, 'current(A)': current_interp})
        
        df_interp_cc = df_interp[df_interp['time(s)'] <= t_cc_end]
        df_interp_cv = df_interp[df_interp['time(s)'] > t_cc_end]
        
        return df_interp_cc, df_interp_cv
    except (IndexError, ValueError):
        logging.debug("Interpolation failed due to IndexError or ValueError.", exc_info=True)
        return None

def _calculate_final_features(df_interp_cc: pd.DataFrame, df_interp_cv: pd.DataFrame, capacity: float, args: argparse.Namespace) -> dict | None:
    """Helper function to calculate all statistics on the separated CC and CV data."""
    v_max_cc = np.max(df_interp_cc['voltage(v)'])
    feature_window_v = df_interp_cc[df_interp_cc['voltage(v)'] >= (v_max_cc - args.voltage_window)]
    feature_window_c = df_interp_cv[(df_interp_cv['current(A)'] >= args.current_min) & (df_interp_cv['current(A)'] <= args.current_max)]

    if len(feature_window_v) < 2 or len(feature_window_c) < 2: return None

    voltage_stats = calculate_statistics(feature_window_v['voltage(v)'], feature_window_v['time(s)'], args.hist_bins)
    current_stats = calculate_statistics(feature_window_c['current(A)'], feature_window_c['time(s)'], args.hist_bins)
    
    cc_q = simpson(y=feature_window_v['current(A)'], x=feature_window_v['time(s)']) / 3600
    cv_q = simpson(y=feature_window_c['current(A)'], x=feature_window_c['time(s)']) / 3600
    cc_charge_time = feature_window_v['time(s)'].iloc[-1] - feature_window_v['time(s)'].iloc[0]
    cv_charge_time = feature_window_c['time(s)'].iloc[-1] - feature_window_c['time(s)'].iloc[0]

    return {
        'voltage mean': voltage_stats.get('mean'), 'voltage std': voltage_stats.get('std'),
        'voltage skewness': voltage_stats.get('skewness'), 'voltage kurtosis': voltage_stats.get('kurtosis'),
        'CC Q': cc_q, 'CC charge time': cc_charge_time,
        'voltage slope': voltage_stats.get('slope'), 'voltage entropy': voltage_stats.get('entropy'),
        'current mean': current_stats.get('mean'), 'current std': current_stats.get('std'),
        'current skewness': current_stats.get('skewness'), 'current kurtosis': current_stats.get('kurtosis'),
        'CV Q': cv_q, 'CV charge time': cv_charge_time,
        'current slope': current_stats.get('slope'), 'current entropy': current_stats.get('entropy'),
        'capacity': capacity
    }

def load_battery_data(mat_path: Path) -> np.ndarray | None:
    if not mat_path.exists():
        logging.warning(f"Data file not found at {mat_path}. Skipping.")
        return None
    try:
        raw_data = scipy.io.loadmat(mat_path)
        return safe_get(raw_data, [mat_path.stem, 0, 0, 'cycle', 0])
    except Exception:
        logging.error(f"Failed to load or parse .mat file {mat_path}.", exc_info=True)
        return None

def get_discharge_capacity(cycle_index: int, all_cycles: np.ndarray) -> float | None:
    if cycle_index + 1 < len(all_cycles):
        next_cycle = all_cycles[cycle_index + 1]
        cycle_type = safe_get(next_cycle, ['type', 0])
        
        if cycle_type == 'discharge':
            return safe_get(next_cycle, ['data', 0, 0, 'Capacity', 0, 0])
        
        if cycle_type == 'impedance' and cycle_index + 2 < len(all_cycles):
            cycle_after_impedance = all_cycles[cycle_index + 2]
            if safe_get(cycle_after_impedance, ['type', 0]) == 'discharge':
                return safe_get(cycle_after_impedance, ['data', 0, 0, 'Capacity', 0, 0])
    return None

def calculate_statistics(data_series: pd.Series, time_series: pd.Series, hist_bins: int) -> dict:
    if len(data_series) < 2: return {}
    hist_probs, _ = np.histogram(data_series, bins=hist_bins, density=True)
    return {
        'mean': np.mean(data_series), 'std': np.std(data_series),
        'skewness': skew(data_series), 'kurtosis': kurtosis(data_series),
        'slope': (data_series.iloc[-1] - data_series.iloc[0]) / (time_series.iloc[-1] - time_series.iloc[0]),
        'entropy': entropy(hist_probs + 1e-12)
    }

def extract_cycle_features(cycle_metadata: dict, capacity: float, cycle_num: int, battery_name: str, args: argparse.Namespace) -> dict | None:
    data_dict = safe_get(cycle_metadata, ['data', 0, 0])
    if data_dict is None: return None

    try:
        time_data = safe_get(data_dict, ['Time', 0], default=np.array([]))
        voltage_data = safe_get(data_dict, ['Voltage_measured', 0], default=np.array([]))
        current_data = safe_get(data_dict, ['Current_measured', 0], default=np.array([]))

        df = pd.DataFrame({
            'time(s)': time_data.flatten(),
            'voltage(v)': voltage_data.flatten(),
            'current(A)': current_data.flatten()
        })
        df.dropna(inplace=True)
        if len(df) < 4: return None

        interpolation_result = _perform_interpolation(df, args.time_step, args.voltage_threshold)
        if interpolation_result is None:
            logging.warning(f"Skipping cycle {cycle_num} for {battery_name}: Interpolation failed.")
            return None
        
        df_interp_cc, df_interp_cv = interpolation_result
        features = _calculate_final_features(df_interp_cc, df_interp_cv, capacity, args)
        if features is None:
            logging.warning(f"Skipping cycle {cycle_num} for {battery_name}: Feature calculation failed.")
            return None
        
        return features
    except Exception:
        logging.error(f"An unexpected error occurred processing cycle {cycle_num} for {battery_name}.", exc_info=True)
        return None

def main():
    script_dir = Path(__file__).parent
    # The default data directory is now relative to the script's location
    default_data_dir = script_dir / "data" / "NASA data" / "data"

    parser = argparse.ArgumentParser(
        description="Extract features from NASA battery dataset .mat files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-d', '--data-directory', type=str, default=default_data_dir, help="Path to the directory containing the .mat files.")
    parser.add_argument('--log-file', type=str, default="feature_extraction.log", help="Name of the log file.")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging verbosity level.')
    parser.add_argument('--time-step', type=float, default=2.5, help="Time step (dt) for interpolation.")
    parser.add_argument('--voltage-threshold', type=float, default=4.2, help="Voltage to determine CC/CV transition.")
    parser.add_argument('--voltage-window', type=float, default=0.1, help="Voltage window for CC feature analysis (V_max - V_window).")
    parser.add_argument('--current-min', type=float, default=0.1, help="Minimum current for CV feature analysis window.")
    parser.add_argument('--current-max', type=float, default=0.5, help="Maximum current for CV feature analysis window.")
    parser.add_argument('--hist-bins', type=int, default=50, help="Number of bins for entropy calculation histogram.")

    args = parser.parse_args()
    
    base_dir = Path(args.data_directory)
    setup_logging(args.log_level, base_dir / args.log_file)
    
    if not base_dir.is_dir():
        logging.critical(f"Error: The specified directory does not exist: {base_dir}")
        return

    battery_files = sorted(base_dir.glob('B????.mat'))
    if not battery_files:
        logging.warning(f"No battery data files (e.g., B0005.mat) found in {base_dir}")
        return

    logging.info(f"Starting feature extraction for {len(battery_files)} batteries with parameters: {vars(args)}")
    for mat_path in battery_files:
        battery_id = mat_path.stem
        logging.info(f"--- Processing Battery: {battery_id} ---")
        
        all_cycle_data = load_battery_data(mat_path)
        if all_cycle_data is None: continue

        all_features = []
        for i, cycle_metadata in enumerate(all_cycle_data):
            if safe_get(cycle_metadata, ['type', 0]) == 'charge':
                capacity = get_discharge_capacity(i, all_cycle_data)
                if capacity is not None:
                    features = extract_cycle_features(cycle_metadata, capacity, i + 1, battery_id, args)
                    if features:
                        all_features.append(features)
        
        if all_features:
            features_df = pd.DataFrame(all_features)
            # Check if all columns exist before reordering to prevent KeyErrors
            final_columns = [col for col in COLUMN_ORDER if col in features_df.columns]
            features_df = features_df[final_columns]

            output_filename = f'{battery_id}.csv'
            output_path = base_dir / output_filename
            features_df.to_csv(output_path, index=False)
            logging.info(f"Successfully saved features for {battery_id} to {output_path}\n")
        else:
            logging.warning(f"No features were extracted for Battery {battery_id}.\n")
    logging.info("--- Feature extraction process completed. ---")

if __name__ == "__main__":
    main()