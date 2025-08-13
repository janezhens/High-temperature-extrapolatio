import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import gmean
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class PH3DataAnalyzer:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.datasets = {}
        self.results = {}
        
    def load_data(self):
        """加载所有PH3数据文件"""
        for file_path in self.file_paths:
            # 从文件名提取振动态信息
            if 'v=1' in file_path:
                key = 'PH3_v=1'
            elif 'v=10' in file_path:
                key = 'PH3_v=10'
            elif 'v=20' in file_path:
                key = 'PH3_v=20'
            else:
                key = file_path.split('/')[-1].replace('.xlsm', '')
            
            try:
                df = pd.read_excel(file_path)
                self.datasets[key] = df
                print(f"成功加载 {key}: {df.shape[0]} 行, {df.shape[1]} 列")
            except Exception as e:
                print(f"加载 {file_path} 时出错: {e}")
    
    def basic_info_comparison(self):
        """基本信息对比"""
        print("\n=== PH3数据基本信息对比 ===")
        info_data = []
        
        for name, df in self.datasets.items():
            temp_cols = [col for col in df.columns if col.startswith('t=')]
            info_data.append({
                '数据集': name,
                '数据点数': len(df),
                '温度点数': len(temp_cols),
                '波数范围': f"{df['wavenumber'].min():.0f} - {df['wavenumber'].max():.0f}",
                '温度范围': f"{temp_cols[0]} - {temp_cols[-1]}",
                '数据范围': f"{df[temp_cols].min().min():.2e} - {df[temp_cols].max().max():.2e}"
            })
        
        info_df = pd.DataFrame(info_data)
        print(info_df.to_string(index=False))
        return info_df
    
    def preprocess_data(self, df, test_temps=['t=1800k', 't=1900k', 't=2000k']):
        """数据预处理和标准化"""
        # 获取可用的测试温度列
        available_test_temps = [col for col in test_temps if col in df.columns]
        if not available_test_temps:
            # 如果没有指定的测试温度，使用最后几个温度点
            temp_cols = [col for col in df.columns if col.startswith('t=')]
            available_test_temps = temp_cols[-3:] if len(temp_cols) >= 3 else temp_cols[-1:]
        
        # 训练数据列
        train_cols = [col for col in df.columns if col.startswith('t=') and col not in available_test_temps]
        
        if not train_cols:
            print(f"警告: 没有足够的训练数据列")
            return None, None, None, None
        
        # 提取训练数据
        train_data = df[train_cols].copy()
        
        # 按行标准化
        row_means = train_data.mean(axis=1)
        row_stds = train_data.std(axis=1)
        
        # 避免除零
        row_stds = row_stds.replace(0, 1e-10)
        
        # 标准化训练数据
        train_scaled = (train_data.subtract(row_means, axis=0)).divide(row_stds, axis=0)
        
        # 标准化测试数据
        if available_test_temps:
            test_data = df[available_test_temps].copy()
            test_scaled = (test_data.subtract(row_means, axis=0)).divide(row_stds, axis=0)
        else:
            test_scaled = pd.DataFrame()
        
        # 创建最终的训练和测试数据框
        train_df = pd.concat([df[['wavenumber']], train_scaled], axis=1)
        test_df = pd.concat([df[['wavenumber']], test_scaled], axis=1) if not test_scaled.empty else pd.DataFrame()
        
        return train_df, test_df, row_means, row_stds
    
    def prepare_ml_data(self, train_df, test_df):
        """准备机器学习数据"""
        if train_df is None or train_df.empty:
            return None, None, None, None
        
        # 转换为长格式
        train_long = pd.melt(
            train_df,
            id_vars='wavenumber',
            value_vars=[col for col in train_df.columns if col.startswith('t=')],
            var_name='temperature',
            value_name='cross_section'
        )
        
        # 清理温度列
        train_long['temperature'] = train_long['temperature'].str.replace('t=', '', regex=False).str.replace('k', '', regex=False).astype(int)
        
        X_train = train_long[['wavenumber', 'temperature']]
        y_train = train_long['cross_section']
        
        # 处理测试数据
        if test_df is not None and not test_df.empty:
            test_long = pd.melt(
                test_df,
                id_vars='wavenumber',
                value_vars=[col for col in test_df.columns if col.startswith('t=')],
                var_name='temperature',
                value_name='cross_section'
            )
            
            test_long['temperature'] = test_long['temperature'].str.replace('t=', '', regex=False).str.replace('k', '', regex=False).astype(int)
            
            X_test = test_long[['wavenumber', 'temperature']]
            y_test = test_long['cross_section']
        else:
            X_test, y_test = None, None
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train, y_train, degree=6, alpha=1.0):
        """训练多项式回归模型"""
        model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            Lasso(alpha=alpha, max_iter=5000)
        )
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        if X_test is None or y_test is None:
            return None
        
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 计算MAPE，避免除零
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        # 计算几何平均误差
        gme = gmean(np.abs(y_test - y_pred) + 1e-8)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'GME': gme,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def analyze_all_datasets(self):
        """分析所有数据集"""
        print("\n=== 开始机器学习分析 ===")
        
        for name, df in self.datasets.items():
            print(f"\n分析 {name}...")
            
            # 数据预处理
            train_df, test_df, row_means, row_stds = self.preprocess_data(df)
            
            if train_df is None:
                print(f"跳过 {name}: 数据预处理失败")
                continue
            
            # 准备机器学习数据
            X_train, y_train, X_test, y_test = self.prepare_ml_data(train_df, test_df)
            
            if X_train is None:
                print(f"跳过 {name}: 机器学习数据准备失败")
                continue
            
            # 训练模型
            model = self.train_model(X_train, y_train)
            
            # 评估模型
            if X_test is not None:
                evaluation = self.evaluate_model(model, X_test, y_test)
                if evaluation:
                    self.results[name] = evaluation
                    print(f"{name} 模型评估完成")
                    print(f"  R² = {evaluation['R2']:.4f}")
                    print(f"  RMSE = {evaluation['RMSE']:.4e}")
                    print(f"  MAPE = {evaluation['MAPE']:.2f}%")
            else:
                print(f"{name}: 无测试数据，跳过评估")
    
    def plot_comparison_results(self):
        """绘制对比结果"""
        if not self.results:
            print("没有可用的结果进行绘图")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PH3 振动态对比分析结果', fontsize=16, fontweight='bold')
        
        # 1. 预测vs真实值散点图
        ax1 = axes[0, 0]
        colors = ['blue', 'red', 'green']
        for i, (name, result) in enumerate(self.results.items()):
            y_test = result['y_test']
            y_pred = result['y_pred']
            ax1.scatter(y_test, y_pred, alpha=0.6, s=20, color=colors[i % len(colors)], label=name)
        
        # 添加y=x线
        all_y = np.concatenate([result['y_test'] for result in self.results.values()])
        min_val, max_val = all_y.min(), all_y.max()
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.set_title('预测值 vs 真实值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 模型性能指标对比
        ax2 = axes[0, 1]
        metrics = ['R2', 'RMSE', 'MAE', 'MAPE']
        x_pos = np.arange(len(self.results))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in self.results.keys()]
            ax2.bar(x_pos + i * width, values, width, label=metric, alpha=0.8)
        
        ax2.set_xlabel('数据集')
        ax2.set_ylabel('指标值')
        ax2.set_title('模型性能指标对比')
        ax2.set_xticks(x_pos + width * 1.5)
        ax2.set_xticklabels(list(self.results.keys()), rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差分布
        ax3 = axes[1, 0]
        for i, (name, result) in enumerate(self.results.items()):
            residuals = result['y_test'] - result['y_pred']
            ax3.hist(residuals, bins=50, alpha=0.6, label=name, color=colors[i % len(colors)])
        
        ax3.set_xlabel('残差')
        ax3.set_ylabel('频数')
        ax3.set_title('残差分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 数据集基本信息对比
        ax4 = axes[1, 1]
        info_data = []
        for name, df in self.datasets.items():
            temp_cols = [col for col in df.columns if col.startswith('t=')]
            info_data.append([name, len(df), len(temp_cols)])
        
        info_array = np.array(info_data)
        x_pos = np.arange(len(info_data))
        
        ax4_twin = ax4.twinx()
        bars1 = ax4.bar(x_pos - 0.2, [int(x) for x in info_array[:, 1]], 0.4, label='数据点数', alpha=0.8, color='skyblue')
        bars2 = ax4_twin.bar(x_pos + 0.2, [int(x) for x in info_array[:, 2]], 0.4, label='温度点数', alpha=0.8, color='lightcoral')
        
        ax4.set_xlabel('数据集')
        ax4.set_ylabel('数据点数', color='skyblue')
        ax4_twin.set_ylabel('温度点数', color='lightcoral')
        ax4.set_title('数据集规模对比')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([x[0] for x in info_data], rotation=45)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'{int(height)}', 
                    ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax4_twin.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'{int(height)}', 
                         ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('ph3_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*60)
        print("PH3 振动态数据分析总结报告")
        print("="*60)
        
        # 基本信息
        print("\n1. 数据集概况:")
        for name, df in self.datasets.items():
            temp_cols = [col for col in df.columns if col.startswith('t=')]
            print(f"   {name}: {len(df)}个数据点, {len(temp_cols)}个温度点")
        
        # 模型性能
        if self.results:
            print("\n2. 模型性能对比:")
            performance_data = []
            for name, result in self.results.items():
                performance_data.append({
                    '数据集': name,
                    'R²': f"{result['R2']:.4f}",
                    'RMSE': f"{result['RMSE']:.4e}",
                    'MAE': f"{result['MAE']:.4e}",
                    'MAPE(%)': f"{result['MAPE']:.2f}",
                    'GME': f"{result['GME']:.4e}"
                })
            
            performance_df = pd.DataFrame(performance_data)
            print(performance_df.to_string(index=False))
            
            # 找出最佳模型
            best_r2 = max(self.results.items(), key=lambda x: x[1]['R2'])
            best_rmse = min(self.results.items(), key=lambda x: x[1]['RMSE'])
            
            print(f"\n3. 最佳性能:")
            print(f"   最高R²: {best_r2[0]} (R² = {best_r2[1]['R2']:.4f})")
            print(f"   最低RMSE: {best_rmse[0]} (RMSE = {best_rmse[1]['RMSE']:.4e})")
        
        # 保存结果到Excel
        if self.results:
            with pd.ExcelWriter('ph3_analysis_results.xlsx', engine='openpyxl') as writer:
                # 基本信息
                info_df = self.basic_info_comparison()
                info_df.to_excel(writer, sheet_name='基本信息', index=False)
                
                # 性能指标
                performance_df.to_excel(writer, sheet_name='性能指标', index=False)
                
                print(f"\n结果已保存到 ph3_analysis_results.xlsx")
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("开始PH3数据综合分析...")
        
        # 加载数据
        self.load_data()
        
        if not self.datasets:
            print("没有成功加载任何数据集")
            return
        
        # 基本信息对比
        self.basic_info_comparison()
        
        # 机器学习分析
        self.analyze_all_datasets()
        
        # 绘制对比图
        self.plot_comparison_results()
        
        # 生成总结报告
        self.generate_summary_report()
        
        print("\nPH3数据分析完成！")

# 主程序
if __name__ == "__main__":
    # PH3数据文件路径
    ph3_files = [
        '/Users/shuaiyuanzhen/Desktop/High temperature extrapolation/ph3_v=1.xlsm',
        '/Users/shuaiyuanzhen/Desktop/High temperature extrapolation/ph3_v=10.xlsm',
        '/Users/shuaiyuanzhen/Desktop/High temperature extrapolation/ph3_v=20.xlsm'
    ]
    
    # 创建分析器并运行分析
    analyzer = PH3DataAnalyzer(ph3_files)
    analyzer.run_complete_analysis()