# -*- coding: utf-8 -*-
# 1. Standard Library Imports
import argparse
import os
import time
import warnings

# 2. Third-Party Imports
import lightning as pl
import matplotlib
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE
from sklearn.metrics import r2_score

# 3. Local Application/Library Specific Imports
from assistant import get_gpus_memory_info, print_log, set_seed
from Helper_Plot import single_model_draw_test_CY25_1_plt
from ModelsModify.MambaSimple import MambaSimpleNetModel
from NASADataPreProcess import MultiVariateBatteryDataProcess

# --- Initial Setup ---
matplotlib.use('agg')
warnings.filterwarnings("ignore")

# --- Helper Functions ---

def rul_value_error(y_test, y_predict, threshold):
    """Calculates the RUL error and related metrics."""
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    rul_real = true_re + 1
    rul_pred = pred_re + 1
    ae_error = abs(true_re - pred_re)
    re_score = abs(true_re - pred_re) / true_re if true_re > 0 else float('inf')
    if re_score > 1: re_score = 1
    return rul_real, rul_pred, ae_error, re_score

def setup_arg_parser():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="MambaSimple RUL Prediction")
    parser.add_argument('--model', default='MambaSimple', help='Model name.')
    parser.add_argument('--seq_len', type=int, default=20, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--test_name', type=str, default='B0005', help='Battery data used for test')
    parser.add_argument('--start_point_list', nargs='+', type=int, default=[50, 70, 90], help='Cycle start points')
    parser.add_argument('--count', type=int, default=10, help='Number of independent experiments.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--max_epochs', type=int, default=200, help='Max training epochs.')
    parser.add_argument('--train_split_ratio', type=float, default=0.8, help='Train/validation split ratio.')
    parser.add_argument('--rul_threshold_ratio', type=float, default=0.7, help='EOL threshold ratio.')
    parser.add_argument('--Rated_Capacity', type=float, default=2.0, help='Rated Capacity')
    parser.add_argument('--root_dir', type=str, default='PerioMamba_RUL_prediction_sl_20', help='Root for results.')
    return parser.parse_args()

def prepare_dataloaders(df_train, df_test, args):
    """Creates and returns dataloaders for training, validation, and testing."""
    train_cutoff = int(len(df_train) * args.train_split_ratio)
    
    time_varying_known_reals = [
        'Cycle', 'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy',
        'monotonic_trend', 'residual_noise'
    ]
    
    dataset_params = {
        "time_idx": "time_idx",
        "target": "target",
        "group_ids": ['group_id'],
        "max_encoder_length": args.seq_len,
        "max_prediction_length": args.pred_len,
        "time_varying_known_reals": time_varying_known_reals,
        "time_varying_unknown_reals": ['target'],
        "target_normalizer": EncoderNormalizer(),
        "add_encoder_length": False,
    }

    # 1. Create the training dataset first. All normalizers are fitted on this data.
    training_dataset = TimeSeriesDataSet(df_train[:train_cutoff], **dataset_params)

    # 2. Create validation and testing sets FROM the training dataset.
    # This ensures they inherit the fitted normalizers and encoders, preventing data leakage.
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, df_train[train_cutoff:], stop_randomization=True)
    testing_dataset = TimeSeriesDataSet.from_dataset(training_dataset, df_test, stop_randomization=True)

    # Create dataloaders from the datasets
    train_loader = training_dataset.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0, drop_last=True)
    val_loader = validation_dataset.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)
    test_loader = testing_dataset.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    return train_loader, val_loader, test_loader, training_dataset, time_varying_known_reals

def run_single_experiment(exp_count, training_dataset, dataloaders, time_varying_known_reals, save_dir, args):
    """
    Runs a single training and evaluation cycle.
    Note: The 'dataloaders' parameter should be a tuple of (train, val, test) loaders.
    """
    train_loader, val_loader, test_loader = dataloaders
    set_seed(exp_count)

    model = MambaSimpleNetModel.from_dataset(
        training_dataset,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        enc_in=len(time_varying_known_reals),
        dec_in=len(time_varying_known_reals),
        c_out=1,
        learning_rate=0.0022,
        loss=SMAPE(), 
    )

    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, verbose=False, mode='min')
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=0.2,
        callbacks=[early_stop_callback],
        logger=False,
        default_root_dir=save_dir,
    )

    start_time = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_time = time.time() - start_time

    best_model = MambaSimpleNetModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    start_time = time.time()
    predictions = best_model.predict(test_loader, batch_size=256)
    infer_time = time.time() - start_time
    
    # Return raw predictions and performance metrics
    return predictions.detach().cpu().numpy().reshape(-1), {
        "train_time": train_time, "infer_time": infer_time, 
        "epochs": trainer.current_epoch
    }

def calculate_and_log_run_metrics(y_true, y_pred, perf_metrics, args, exp_count, log_files):
    """Calculates metrics for a single run and logs them."""
    stat_log, exp_log = log_files
    
    # Calculate performance metrics
    metrics = {}
    metrics["MAE"] = np.mean(np.abs(y_true - y_pred))
    metrics["RMSE"] = np.sqrt(np.mean(np.square(y_true - y_pred)))
    metrics["r2"] = r2_score(y_true, y_pred)
    
    RUL_real, RUL_pred, AE, RE = rul_value_error(y_true, y_pred, args.Rated_Capacity * args.rul_threshold_ratio)
    metrics.update({"RUL_real": RUL_real, "RUL_pred": RUL_pred, "AE": AE, "RE": RE})
    metrics.update(perf_metrics)
    
    # Log the results
    log_line = (f"Run {exp_count}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['r2']:.4f}, "
                f"RUL_real={metrics['RUL_real']}, RUL_pred={metrics['RUL_pred']}, AE={metrics['AE']}, RE={metrics['RE']:.4f}, "
                f"Epochs={metrics['epochs']}, TrainTime={metrics['train_time']:.2f}s, InferTime={metrics['infer_time']:.2f}s")
    print_log(log_line, stat_log)
    print_log(log_line, exp_log)
    
    return metrics

def log_average_metrics(metrics_summary, log_file):
    """Calculates and logs the average of metrics over all runs."""
    print_log("\n--- Average Metrics ---", log_file)
    for key, values in metrics_summary.items():
        avg_val = np.mean(values)
        print_log(f"Average {key}: {avg_val:.4f}", log_file)

def main():
    """Main execution function."""
    args = setup_arg_parser()
    
    try:
        gpu_id, _ = get_gpus_memory_info()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Automatically selected GPU: {gpu_id}")
    except Exception as e:
        print(f"Could not set GPU automatically: {e}. Defaulting to GPU 0.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    BatteryData = np.load('data/NASA data/NASA_dataset.npy', allow_pickle=True).item()

    root_dir = f'results_{args.root_dir}/{args.test_name}/{args.model}/'
    save_figure_dir = os.path.join(root_dir, 'figures')
    os.makedirs(save_figure_dir, exist_ok=True)

    all_start_points_predictions = {}
    
    _, _, df_all = MultiVariateBatteryDataProcess(BatteryData, args.test_name, args.start_point_list[0], args)
    real_data = df_all['target'].values * args.Rated_Capacity

    for start_point in args.start_point_list:
        print(f"\n----- Processing Start Point: {start_point} for Battery: {args.test_name} -----")
        
        sp_root_dir = os.path.join(root_dir, f'SP{start_point}/')
        os.makedirs(sp_root_dir, exist_ok=True)
        stat_log_path = os.path.join(sp_root_dir, 'log_stat.txt')
        stat_log = open(stat_log_path, 'w', encoding='UTF-8')

        print_log(f'Model: {args.model}\nBattery: {args.test_name}, Start Point: {start_point}\n', stat_log)

        df_train, df_test, _ = MultiVariateBatteryDataProcess(BatteryData, args.test_name, start_point, args)
        
        # --- THIS IS THE FIX (PART 1) ---
        # Unpack all four return values from prepare_dataloaders into distinct variables
        train_loader, val_loader, test_loader, training_dataset, time_varying_known_reals= prepare_dataloaders(df_train, df_test, args)
        
        metrics_summary = {"MAE": [], "RMSE": [], "r2": [], "RE": [], "AE": [], "train_time": [], "infer_time": [], "epochs": []}
        predictions_per_run = []

        for i in range(1, args.count + 1):
            exp_save_dir = os.path.join(sp_root_dir, f'Exp{i}/')
            os.makedirs(exp_save_dir, exist_ok=True)
            exp_log_path = os.path.join(exp_save_dir, 'log_exp.txt')
            exp_log = open(exp_log_path, 'w', encoding='UTF-8')

            # --- THIS IS THE FIX (PART 2) ---
            # Pass the correct variables to the function. Note how `dataloaders` is now a 3-item tuple.
            raw_predictions, perf_metrics = run_single_experiment(
                exp_count=i,
                training_dataset=training_dataset,
                dataloaders=(train_loader, val_loader, test_loader),
                time_varying_known_reals=time_varying_known_reals,
                save_dir=exp_save_dir,
                args=args
            )
            
            y_pred = raw_predictions * args.Rated_Capacity
            predictions_per_run.append(y_pred)

            actuals_df = df_all.loc[df_all['Cycle'] >= start_point, 'target']
            y_true = actuals_df.values * args.Rated_Capacity

            NMAE = np.mean(np.abs(y_true - y_pred))
            NRMSE = np.sqrt(np.mean(np.square(y_true - y_pred)))
            r2 = r2_score(y_true, y_pred)
            RUL_real, RUL_pred, AE, RE = rul_value_error(y_true, y_pred, args.Rated_Capacity * args.rul_threshold_ratio)
            
            log_line = (f"Run {i}: MAE={NMAE:.4f}, RMSE={NRMSE:.4f}, R2={r2:.4f}, RUL_real={RUL_real}, "
                        f"RUL_pred={RUL_pred}, AE={AE}, RE={RE:.4f}, Epochs={perf_metrics['epochs']}, "
                        f"TrainTime={perf_metrics['train_time']:.2f}s, InferTime={perf_metrics['infer_time']:.2f}s")
            print_log(log_line, stat_log)
            print_log(log_line, exp_log)
            exp_log.close()
            
            # Aggregate metrics
            metrics_summary["MAE"].append(NMAE)
            metrics_summary["RMSE"].append(NRMSE)
            metrics_summary["r2"].append(r2)
            metrics_summary["RE"].append(RE)
            metrics_summary["AE"].append(AE)
            metrics_summary["train_time"].append(perf_metrics['train_time'])
            metrics_summary["infer_time"].append(perf_metrics['infer_time'])
            metrics_summary["epochs"].append(perf_metrics['epochs'])

        # Log average metrics
        print_log("\n--- Average Metrics ---", stat_log)
        for key, values in metrics_summary.items():
            avg_val = np.mean(values)
            print_log(f"Average {key}: {avg_val:.4f}", stat_log)
        stat_log.close()

        all_start_points_predictions[f'SP{start_point}'] = predictions_per_run

    # Save all prediction results
    results_path = os.path.join('results', f'RUL_{args.test_name}_{args.model}.pth')
    os.makedirs('results', exist_ok=True)
    torch.save(all_start_points_predictions, results_path)
    print(f"\nAll predictions saved to {results_path}")

    # Plotting results for each run if it's battery B0005
    if args.test_name == 'B0005' and args.count > 0:
        print(f"\nGenerating {args.count} result plot(s)...")
        for i in range(args.count):
            save_filename = f'best_model_run_{i+1}_RUL_Prediction'
            single_model_draw_test_CY25_1_plt(
                real_data,
                all_start_points_predictions['SP50'][i],
                all_start_points_predictions['SP70'][i],
                all_start_points_predictions['SP90'][i],
                save_filename=save_filename,
                save_figure_dir=save_figure_dir,
                Rated_Capacity=args.Rated_Capacity,
                model=args.model
            )
        print(f"All plots saved to {save_figure_dir}")


if __name__ == "__main__":
    main()