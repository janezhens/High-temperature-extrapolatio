#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2O Cross Section Prediction Visualization
Comprehensive visualization of H2O prediction results vs actual data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set font for plots
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """
    Load H2O prediction results and original data
    """
    print("Loading H2O prediction results...")
    
    # Load prediction results
    df_pred = pd.read_csv('H2O_cross_section_predictionss.csv')
    print(f"Prediction data shape: {df_pred.shape}")
    
    # Load original H2O data
    df_h2o_original = pd.read_excel('h2o_output1.xlsm')
    print(f"Original H2O data shape: {df_h2o_original.shape}")
    
    return df_pred, df_h2o_original

def create_comprehensive_visualization(df_pred, df_h2o_original):
    """
    Create comprehensive visualization charts
    """
    print("Creating comprehensive visualization...")
    
    # Create large figure
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('H2O Cross Section Prediction Results Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Heatmap comparison
    create_heatmap_comparison(df_pred, df_h2o_original, fig, 1)
    
    # 2. Temperature comparison
    create_temperature_comparison(df_pred, fig, 2)
    
    # 3. Scatter plot comparison
    create_scatter_comparison(df_pred, fig, 3)
    
    # 4. Error analysis
    create_error_analysis(df_pred, fig, 4)
    
    # 5. Statistical metrics
    create_statistical_metrics(df_pred, fig, 5)
    
    # 6. Residual analysis
    create_residual_analysis(df_pred, fig, 6)
    
    # 7. Temperature error analysis
    create_temperature_error_analysis(df_pred, fig, 7)
    
    # 8. Wavenumber error analysis
    create_wavenumber_error_analysis(df_pred, fig, 8)
    
    # 9. Prediction vs actual value distribution
    create_prediction_distribution(df_pred, fig, 9)
    
    plt.tight_layout()
    plt.savefig('H2O_prediction_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'H2O_prediction_visualizations.png'")
    plt.show()

def create_heatmap_comparison(df_pred, df_h2o_original, fig, subplot_num):
    """
    Create heatmap comparison - Fixed data shape issues
    """
    ax = fig.add_subplot(3, 3, subplot_num)
    
    # Select several temperature points for comparison
    selected_temps = [100, 500, 1000, 1500, 2000, 2500]
    temp_labels = ['100K', '500K', '1000K', '1500K', '2000K', '2500K']
    
    # Prepare data - Ensure all temperatures have data of the same length
    wavenumbers = df_pred['wavenumber'].unique()
    wavenumbers = sorted(wavenumbers)[::10]  # Take one point every 10
    
    # Create comparison matrix
    comparison_matrix = np.zeros((len(selected_temps), len(wavenumbers)))
    
    for i, temp in enumerate(selected_temps):
        temp_data = df_pred[df_pred['temperature'] == temp]
        temp_data = temp_data[temp_data['wavenumber'].isin(wavenumbers)]
        
        if len(temp_data) > 0:
            # Ensure data is sorted by wavenumber
            temp_data = temp_data.sort_values('wavenumber')
            
            # Calculate relative error
            mask = (temp_data['actual_cross_section'] > 0) & (temp_data['predicted_cross_section'] > 0)
            if mask.sum() > 0:
                relative_error = np.abs(temp_data.loc[mask, 'actual_cross_section'] - 
                                      temp_data.loc[mask, 'predicted_cross_section']) / temp_data.loc[mask, 'actual_cross_section']
                
                # Fill error values to corresponding positions
                for j, (_, row) in enumerate(temp_data[mask].iterrows()):
                    if row['wavenumber'] in wavenumbers:
                        wavenumber_idx = list(wavenumbers).index(row['wavenumber'])
                        comparison_matrix[i, wavenumber_idx] = relative_error.iloc[j]
    
    # Create heatmap
    im = ax.imshow(comparison_matrix, cmap='Reds', aspect='auto')
    ax.set_xticks(range(0, len(wavenumbers), 5))
    ax.set_xticklabels([f'{w:.0f}' for w in wavenumbers[::5]], rotation=45)
    ax.set_yticks(range(len(selected_temps)))
    ax.set_yticklabels(temp_labels)
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Temperature')
    ax.set_title('H2O Prediction Relative Error Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Error')

def create_temperature_comparison(df_pred, fig, subplot_num):
    """
    Create comparison of predicted vs actual values at different temperatures
    """
    ax = fig.add_subplot(3, 3, subplot_num)
    
    # Select several temperatures for visualization
    selected_temps = [100]
    colors = ['blue']
    
    for i, temp in enumerate(selected_temps):
        temp_data = df_pred[df_pred['temperature'] == temp]
        
        if len(temp_data) > 0:
            # Filter valid data
            mask = (temp_data['actual_cross_section'] > 0) & (temp_data['predicted_cross_section'] > 0)
            valid_data = temp_data[mask]
            
            if len(valid_data) > 0:
                ax.plot(valid_data['wavenumber'], valid_data['actual_cross_section'], 
                       color='blue', linestyle='-', linewidth=0.5, label=f'{temp}K Actual')
                ax.plot(valid_data['wavenumber'], valid_data['predicted_cross_section'], 
                       color='red', linestyle='--', linewidth=0.5, label=f'{temp}K Predicted')
    
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Cross Section (cm²)')
    ax.set_yscale('log')
    ax.set_title('H2O Cross Section Comparison at Different Temperatures')
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_scatter_comparison(df_pred, fig, subplot_num):
    """
    Create scatter plot comparison
    """
    ax = fig.add_subplot(3, 3, subplot_num)
    
    # Filter valid data
    mask = (df_pred['actual_cross_section'] > 0) & (df_pred['predicted_cross_section'] > 0)
    valid_data = df_pred[mask]
    
    if len(valid_data) > 0:
        # Create scatter plot
        scatter = ax.scatter(valid_data['log_actual'], valid_data['log_predicted'], 
                           c=valid_data['temperature'], cmap='viridis', alpha=0.6, s=20)
        
        # Add diagonal line
        min_val = min(valid_data['log_actual'].min(), valid_data['log_predicted'].min())
        max_val = max(valid_data['log_actual'].max(), valid_data['log_predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction Line')
        
        ax.set_xlabel('Actual Values (log)')
        ax.set_ylabel('Predicted Values (log)')
        ax.set_title('H2O Prediction vs Actual Values Scatter Plot')
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (K)')
        
        # Calculate R²
        r2 = r2_score(valid_data['log_actual'], valid_data['log_predicted'])
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('H2O Prediction vs Actual Values Scatter Plot')

def create_error_analysis(df_pred, fig, subplot_num):
    """
    Create error analysis plot
    """
    ax = fig.add_subplot(3, 3, subplot_num)
    
    # Filter valid data
    mask = (df_pred['actual_cross_section'] > 0) & (df_pred['predicted_cross_section'] > 0)
    valid_data = df_pred[mask]
    
    if len(valid_data) > 0:
        # Calculate errors
        relative_error = np.abs(valid_data['actual_cross_section'] - valid_data['predicted_cross_section']) / valid_data['actual_cross_section']
        absolute_error = np.abs(valid_data['actual_cross_section'] - valid_data['predicted_cross_section'])
        
        # Create error distribution histogram
        ax.hist(relative_error, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Relative Error')
        ax.set_ylabel('Frequency')
        ax.set_title('H2O Prediction Relative Error Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistical information
        mean_error = relative_error.mean()
        median_error = relative_error.median()
        ax.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.3f}')
        ax.axvline(median_error, color='orange', linestyle='--', label=f'Median: {median_error:.3f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('H2O Prediction Relative Error Distribution')

def create_statistical_metrics(df_pred, fig, subplot_num):
    """
    Create statistical metrics table
    """
    ax = fig.add_subplot(3, 3, subplot_num)
    ax.axis('off')
    
    # Filter valid data
    mask = (df_pred['actual_cross_section'] > 0) & (df_pred['predicted_cross_section'] > 0)
    valid_data = df_pred[mask]
    
    if len(valid_data) > 0:
        # Calculate statistical metrics
        mse = mean_squared_error(valid_data['log_actual'], valid_data['log_predicted'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(valid_data['log_actual'], valid_data['log_predicted'])
        r2 = r2_score(valid_data['log_actual'], valid_data['log_predicted'])
        
        relative_error = np.abs(valid_data['actual_cross_section'] - valid_data['predicted_cross_section']) / valid_data['actual_cross_section']
        mean_rel_error = relative_error.mean()
        median_rel_error = relative_error.median()
        
        # Create table
        metrics_data = [
            ['Metric', 'Value'],
            ['MSE', f'{mse:.6f}'],
            ['RMSE', f'{rmse:.6f}'],
            ['MAE', f'{mae:.6f}'],
            ['R²', f'{r2:.6f}'],
            ['Mean Relative Error', f'{mean_rel_error:.6f}'],
            ['Median Relative Error', f'{median_rel_error:.6f}'],
            ['Data Points', f'{len(valid_data)}']
        ]
        
        table = ax.table(cellText=metrics_data[1:], colLabels=metrics_data[0], 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax.set_title('H2O Prediction Statistical Metrics', fontsize=12, fontweight='bold', pad=20)
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('H2O Prediction Statistical Metrics')

def create_residual_analysis(df_pred, fig, subplot_num):
    """
    Create residual analysis plot
    """
    ax = fig.add_subplot(3, 3, subplot_num)
    
    # Filter valid data
    mask = (df_pred['actual_cross_section'] > 0) & (df_pred['predicted_cross_section'] > 0)
    valid_data = df_pred[mask]
    
    if len(valid_data) > 0:
        # Calculate residuals
        residuals = valid_data['log_predicted'] - valid_data['log_actual']
        
        # Create residual plot
        ax.scatter(valid_data['log_actual'], residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Actual Values (log)')
        ax.set_ylabel('Residuals (Predicted - Actual)')
        ax.set_title('H2O Prediction Residual Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add residual statistics
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        ax.text(0.05, 0.95, f'Residual Mean: {mean_residual:.3f}\nResidual Std: {std_residual:.3f}', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('H2O Prediction Residual Analysis')

def create_temperature_error_analysis(df_pred, fig, subplot_num):
    """
    Create temperature error analysis
    """
    ax = fig.add_subplot(3, 3, subplot_num)
    
    # Calculate errors by temperature groups
    temp_errors = []
    temp_labels = []
    
    for temp in sorted(df_pred['temperature'].unique()):
        temp_data = df_pred[df_pred['temperature'] == temp]
        mask = (temp_data['actual_cross_section'] > 0) & (temp_data['predicted_cross_section'] > 0)
        
        if mask.sum() > 0:
            relative_error = np.abs(temp_data.loc[mask, 'actual_cross_section'] - 
                                  temp_data.loc[mask, 'predicted_cross_section']) / temp_data.loc[mask, 'actual_cross_section']
            temp_errors.append(relative_error.mean())
            temp_labels.append(f'{temp}K')
    
    if temp_errors:
        bars = ax.bar(range(len(temp_errors)), temp_errors, color='lightcoral', alpha=0.7)
        ax.set_xticks(range(len(temp_labels)))
        ax.set_xticklabels(temp_labels, rotation=45)
        ax.set_ylabel('Mean Relative Error')
        ax.set_title('H2O Prediction Error vs Temperature')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, error) in enumerate(zip(bars, temp_errors)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{error:.3f}', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('H2O Prediction Error vs Temperature')

def create_wavenumber_error_analysis(df_pred, fig, subplot_num):
    """
    Create wavenumber error analysis
    """
    ax = fig.add_subplot(3, 3, subplot_num)
    
    # Calculate errors by wavenumber groups (every 1000 wavenumbers)
    wavenumber_errors = []
    wavenumber_labels = []
    
    wavenumber_ranges = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
    
    for wmin, wmax in wavenumber_ranges:
        range_data = df_pred[(df_pred['wavenumber'] >= wmin) & (df_pred['wavenumber'] < wmax)]
        mask = (range_data['actual_cross_section'] > 0) & (range_data['predicted_cross_section'] > 0)
        
        if mask.sum() > 0:
            relative_error = np.abs(range_data.loc[mask, 'actual_cross_section'] - 
                                  range_data.loc[mask, 'predicted_cross_section']) / range_data.loc[mask, 'actual_cross_section']
            wavenumber_errors.append(relative_error.mean())
            wavenumber_labels.append(f'{wmin}-{wmax}')
    
    if wavenumber_errors:
        bars = ax.bar(range(len(wavenumber_errors)), wavenumber_errors, color='lightblue', alpha=0.7)
        ax.set_xticks(range(len(wavenumber_labels)))
        ax.set_xticklabels(wavenumber_labels, rotation=45)
        ax.set_ylabel('Mean Relative Error')
        ax.set_title('H2O Prediction Error vs Wavenumber')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, error) in enumerate(zip(bars, wavenumber_errors)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{error:.3f}', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('H2O Prediction Error vs Wavenumber')

def create_prediction_distribution(df_pred, fig, subplot_num):
    """
    Create prediction value distribution plot
    """
    ax = fig.add_subplot(3, 3, subplot_num)
    
    # Filter valid data
    mask = (df_pred['actual_cross_section'] > 0) & (df_pred['predicted_cross_section'] > 0)
    valid_data = df_pred[mask]
    
    if len(valid_data) > 0:
        # Create prediction vs actual value distribution comparison
        ax.hist(valid_data['log_actual'], bins=50, alpha=0.5, label='Actual Values', color='blue')
        ax.hist(valid_data['log_predicted'], bins=50, alpha=0.5, label='Predicted Values', color='red')
        
        ax.set_xlabel('Cross Section (log)')
        ax.set_ylabel('Frequency')
        ax.set_title('H2O Prediction vs Actual Values Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistical information
        actual_mean = valid_data['log_actual'].mean()
        pred_mean = valid_data['log_predicted'].mean()
        ax.axvline(actual_mean, color='blue', linestyle='--', alpha=0.7, label=f'Actual Mean: {actual_mean:.3f}')
        ax.axvline(pred_mean, color='red', linestyle='--', alpha=0.7, label=f'Predicted Mean: {pred_mean:.3f}')
    else:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('H2O Prediction vs Actual Values Distribution')

def main():
    """
    Main function
    """
    print("=== H2O Cross Section Prediction Visualization ===\n")
    
    try:
        # Load data
        df_pred, df_h2o_original = load_and_prepare_data()
        
        # Create visualization
        create_comprehensive_visualization(df_pred, df_h2o_original)
        
        print("\nVisualization completed!")
        print("Generated charts include:")
        print("1. Prediction relative error heatmap")
        print("2. Cross section comparison at different temperatures")
        print("3. Prediction vs actual values scatter plot")
        print("4. Error distribution analysis")
        print("5. Statistical metrics table")
        print("6. Residual analysis")
        print("7. Temperature error analysis")
        print("8. Wavenumber error analysis")
        print("9. Prediction vs actual values distribution comparison")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the H2O prediction script has been run and prediction result files are generated.")

if __name__ == "__main__":
    main() 