import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set font for plots
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SO2DataAnalyzer:
    def __init__(self):
        self.datasets = {}
        self.results = {}
        
    def load_data(self):
        """Load three SO2 datasets"""
        print("Loading datasets...")
        
        file_paths = {
            'SO2_v=1': 'so2_v=1.xlsm',
            'SO2_v=10': 'so2_v=10.xlsm', 
            'SO2_v=20': 'so2_v=20.xlsm'
        }
        
        for name, path in file_paths.items():
            try:
                df = pd.read_excel(path)
                self.datasets[name] = df
                print(f"  ✓ Successfully loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            except FileNotFoundError:
                print(f"  ✗ File not found: {path}")
                return False
            except Exception as e:
                print(f"  ✗ Error loading {path}: {str(e)}")
                return False
        
        return len(self.datasets) > 0
    
    def basic_info_comparison(self):
        """Basic information comparison"""
        print("\n" + "="*60)
        print("Dataset Basic Information Comparison")
        print("="*60)
        
        for name, df in self.datasets.items():
            print(f"\n{name}:")
            print(f"  Data shape: {df.shape}")
            print(f"  Column names: {list(df.columns)}")
            
            # Analyze temperature columns
            temp_cols = [col for col in df.columns if col.startswith('t=')]
            if temp_cols:
                temps = [int(col.replace('t=', '').replace('k', '').replace('K', '')) for col in temp_cols]
                print(f"  Temperature range: {min(temps)}K - {max(temps)}K")
                print(f"  Number of temperature points: {len(temps)}")
            
            # Numerical statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                data_values = df[numeric_cols[1:]].values.flatten()
                data_values = data_values[~np.isnan(data_values)]
                if len(data_values) > 0:
                    print(f"  Value range: {data_values.min():.2e} - {data_values.max():.2e}")
                    print(f"  Mean value: {data_values.mean():.2e}")
                    print(f"  Standard deviation: {data_values.std():.2e}")
    
    def preprocess_data(self):
        """Data preprocessing"""
        print("\n" + "="*60)
        print("Data Preprocessing")
        print("="*60)
        
        for name, df in self.datasets.items():
            print(f"\nProcessing {name}...")
            
            # Check and handle missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                print(f"  Found {missing_count} missing values, using forward fill")
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
            
            # Check infinite values
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                print(f"  Found {inf_count} infinite values, replacing with NaN and filling")
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
            
            print(f"  ✓ Preprocessing completed")
    
    def train_models(self):
        """Train machine learning models"""
        print("\n" + "="*60)
        print("Machine Learning Model Training")
        print("="*60)
        
        for name, df in self.datasets.items():
            print(f"\nTraining {name} model...")
            
            try:
                # Prepare features and targets
                wavenumber_col = df.columns[0]  # First column is wavenumber
                temp_cols = [col for col in df.columns if col.startswith('t=')]
                
                if len(temp_cols) == 0:
                    print(f"  ✗ No temperature columns found")
                    continue
                
                # Create feature matrix
                features_list = []
                targets_list = []
                
                for _, row in df.iterrows():
                    wavenumber = row[wavenumber_col]
                    for temp_col in temp_cols:
                        temp = int(temp_col.replace('t=', '').replace('k', '').replace('K', ''))
                        cross_section = row[temp_col]
                        
                        if not np.isnan(cross_section) and cross_section > 0:
                            features_list.append([wavenumber, temp])
                            targets_list.append(np.log10(cross_section))
                
                if len(features_list) == 0:
                    print(f"  ✗ No valid training data")
                    continue
                
                X = np.array(features_list)
                y = np.array(targets_list)
                
                print(f"  Training data points: {len(X)}")
                
                # Data standardization
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Polynomial features
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X_scaled)
                
                # Split train-test sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X_poly, y, test_size=0.2, random_state=42
                )
                
                # Lasso regression
                model = Lasso(alpha=0.01, max_iter=2000)
                model.fit(X_train, y_train)
                
                # Prediction
                y_pred = model.predict(X_test)
                
                # Calculate evaluation metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # GME (Geometric Mean Error)
                gme = np.exp(np.mean(np.abs(np.log(np.abs(y_test) + 1e-10) - np.log(np.abs(y_pred) + 1e-10))))
                
                # Save results
                self.results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'poly': poly,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'gme': gme
                }
                
                print(f"  ✓ Model training completed")
                print(f"    MSE: {mse:.4e}")
                print(f"    RMSE: {rmse:.4e}")
                print(f"    MAE: {mae:.4e}")
                print(f"    R²: {r2:.4f}")
                print(f"    MAPE: {mape:.2f}%")
                print(f"    GME: {gme:.4e}")
                
            except Exception as e:
                print(f"  ✗ Training failed: {str(e)}")
    
    def create_visualizations(self):
        """Create visualization charts"""
        print("\n" + "="*60)
        print("Generating Visualization Charts")
        print("="*60)
        
        # 1. Prediction results scatter plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SO2 Dataset Prediction Results Comparison', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        for name, result in self.results.items():
            if plot_idx < 3:
                row, col = plot_idx // 2, plot_idx % 2
                ax = axes[row, col]
                
                y_test = result['y_test']
                y_pred = result['y_pred']
                
                ax.scatter(y_test, y_pred, alpha=0.6, s=20)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='y=x')
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{name} (R² = {result["r2"]:.4f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Performance metrics comparison
        if len(self.results) > 1:
            ax = axes[1, 1]
            metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
            x_pos = np.arange(len(metrics))
            width = 0.25
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for i, (name, result) in enumerate(self.results.items()):
                values = [result['mse'], result['rmse'], result['mae'], result['mape']]
                ax.bar(x_pos + i*width, values, width, label=name, color=colors[i % len(colors)])
            
            ax.set_xlabel('Evaluation Metrics')
            ax.set_ylabel('Values')
            ax.set_title('Model Performance Metrics Comparison')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('so2_comparison_results_english.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved prediction results comparison chart: so2_comparison_results_english.png")
        
        # 2. Dataset scale and distribution comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SO2 Dataset Feature Comparison', fontsize=16, fontweight='bold')
        
        # Dataset size comparison
        ax1 = axes[0, 0]
        names = list(self.datasets.keys())
        sizes = [df.shape[0] for df in self.datasets.values()]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax1.bar(names, sizes, color=colors)
        ax1.set_title('Dataset Size Comparison')
        ax1.set_ylabel('Number of Data Points')
        for bar, size in zip(bars, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01, 
                    str(size), ha='center', va='bottom')
        
        # Temperature range comparison
        ax2 = axes[0, 1]
        temp_ranges = []
        for name, df in self.datasets.items():
            temp_cols = [col for col in df.columns if col.startswith('t=')]
            if temp_cols:
                temps = [int(col.replace('t=', '').replace('k', '').replace('K', '')) for col in temp_cols]
                temp_ranges.append((min(temps), max(temps)))
            else:
                temp_ranges.append((0, 0))
        
        x_pos = np.arange(len(names))
        min_temps = [r[0] for r in temp_ranges]
        max_temps = [r[1] for r in temp_ranges]
        
        ax2.bar(x_pos, max_temps, color=colors, alpha=0.7, label='Maximum Temperature')
        ax2.bar(x_pos, min_temps, color=colors, alpha=0.4, label='Minimum Temperature')
        ax2.set_title('Temperature Range Comparison')
        ax2.set_ylabel('Temperature (K)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names)
        ax2.legend()
        
        # Value distribution comparison
        ax3 = axes[1, 0]
        for i, (name, df) in enumerate(self.datasets.items()):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                data_values = df[numeric_cols[1:]].values.flatten()
                data_values = data_values[~np.isnan(data_values)]
                if len(data_values) > 0:
                    ax3.hist(np.log10(data_values + 1e-50), bins=50, alpha=0.6, 
                            label=name, color=colors[i])
        ax3.set_title('Value Distribution Comparison (log10)')
        ax3.set_xlabel('log10(Cross Section Values)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # Residual distribution comparison
        ax4 = axes[1, 1]
        for i, (name, result) in enumerate(self.results.items()):
            residuals = result['y_test'] - result['y_pred']
            ax4.hist(residuals, bins=30, alpha=0.6, label=name, color=colors[i])
        ax4.set_title('Prediction Residual Distribution Comparison')
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('so2_dataset_comparison_english.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved dataset feature comparison chart: so2_dataset_comparison_english.png")
        
        plt.show()
    
    def save_results_to_excel(self):
        """Save results to Excel file"""
        print("\n" + "="*60)
        print("Saving Results")
        print("="*60)
        
        with pd.ExcelWriter('so2_comparison_results_english.xlsx', engine='openpyxl') as writer:
            # Basic information summary
            summary_data = []
            for name, df in self.datasets.items():
                temp_cols = [col for col in df.columns if col.startswith('t=')]
                temps = []
                if temp_cols:
                    temps = [int(col.replace('t=', '').replace('k', '').replace('K', '')) for col in temp_cols]
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                data_values = df[numeric_cols[1:]].values.flatten() if len(numeric_cols) > 1 else []
                data_values = data_values[~np.isnan(data_values)] if len(data_values) > 0 else []
                
                summary_data.append({
                    'Dataset': name,
                    'Data Points': df.shape[0],
                    'Features': df.shape[1],
                    'Min Temperature (K)': min(temps) if temps else 'N/A',
                    'Max Temperature (K)': max(temps) if temps else 'N/A',
                    'Min Value': f"{data_values.min():.2e}" if len(data_values) > 0 else 'N/A',
                    'Max Value': f"{data_values.max():.2e}" if len(data_values) > 0 else 'N/A',
                    'Mean Value': f"{data_values.mean():.2e}" if len(data_values) > 0 else 'N/A'
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Dataset Overview', index=False)
            
            # Model performance comparison
            if self.results:
                performance_data = []
                for name, result in self.results.items():
                    performance_data.append({
                        'Dataset': name,
                        'MSE': f"{result['mse']:.4e}",
                        'RMSE': f"{result['rmse']:.4e}",
                        'MAE': f"{result['mae']:.4e}",
                        'R²': f"{result['r2']:.4f}",
                        'MAPE(%)': f"{result['mape']:.2f}",
                        'GME': f"{result['gme']:.4e}"
                    })
                
                performance_df = pd.DataFrame(performance_data)
                performance_df.to_excel(writer, sheet_name='Model Performance Comparison', index=False)
            
            # Save original data
            for name, df in self.datasets.items():
                sheet_name = f'Raw Data_{name.replace("=", "_")}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print("  ✓ Saved Excel results file: so2_comparison_results_english.xlsx")
    
    def run_complete_analysis(self):
        """Run complete comparative analysis"""
        print("Starting SO2 Dataset Comparative Analysis...")
        print("="*60)
        
        if not self.load_data():
            return
        
        self.basic_info_comparison()
        self.preprocess_data()
        self.train_models()
        self.create_visualizations()
        self.save_results_to_excel()
        
        print("\n" + "="*60)
        print("Analysis Completed!")
        print("="*60)
        print("Generated files:")
        print("  - so2_comparison_results_english.png: Prediction results comparison chart")
        print("  - so2_dataset_comparison_english.png: Dataset feature comparison chart")
        print("  - so2_comparison_results_english.xlsx: Detailed results Excel file")
        print("\nRecommendations:")
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]['r2'])
            print(f"  - Best model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
        print("  - View generated charts to understand dataset differences")
        print("  - Check Excel file for detailed numerical results")

if __name__ == "__main__":
    analyzer = SO2DataAnalyzer()
    analyzer.run_complete_analysis()