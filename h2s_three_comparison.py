import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import gmean
import matplotlib.pyplot as plt

# 定义处理单个数据集的函数
def process_dataset(file_path, dataset_name):
    print(f"\n=== 处理 {dataset_name} 数据集 ===")
    
    # 读取数据
    df = pd.read_excel(file_path)
    print(f"数据形状: {df.shape}")
    
    # 定义要排除的列（测试集）
    columns_to_exclude = ['t=1800k', 't=1900k', 't=2000k']
    columns_to_normalize = [col for col in df.columns if col.startswith('t=') and col not in columns_to_exclude]
    
    print(f"训练温度列数: {len(columns_to_normalize)}")
    print(f"测试温度列数: {len(columns_to_exclude)}")
    
    # Step 1: 提取需要标准化的部分
    data_to_normalize = df[columns_to_normalize].copy()
    
    # Step 2: 按行计算均值和标准差
    row_means = data_to_normalize.mean(axis=1)
    row_stds = data_to_normalize.std(axis=1)
    
    # Step 3: 按行标准化（广播操作）
    df_scaled = (data_to_normalize.subtract(row_means, axis=0)).divide(row_stds, axis=0)
    
    # Step 4: 保存每行均值和标准差
    df_stats = pd.DataFrame({
        'wavenumber': df['wavenumber'],
        'row_mean': row_means,
        'row_std': row_stds
    })
    
    # Step 5: 处理测试集
    test_df = df[columns_to_exclude].copy()
    test_df = (test_df.subtract(row_means, axis=0)).divide(row_stds, axis=0)
    test_df = pd.concat([df[['wavenumber']], test_df], axis=1)
    
    # 训练集
    train_df = pd.concat([df[['wavenumber']], df_scaled], axis=1)
    
    print("标准化完成")
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    return train_df, test_df, df_stats

# 定义建模和评估函数
def train_and_evaluate(train_df, test_df, dataset_name):
    print(f"\n=== {dataset_name} 建模和评估 ===")
    
    # 将宽格式转换成长格式 - 训练集
    long_df = pd.melt(
        train_df,
        id_vars='wavenumber',
        value_vars=[col for col in train_df.columns if col.startswith('t=')],
        var_name='temperature',
        value_name='cross_section'
    )
    
    # 清理 temperature 列，转成整数
    long_df['temperature'] = long_df['temperature'].str.replace('t=', '', regex=False).str.replace('k', '', regex=False).str.replace('K', '', regex=False)
    long_df['temperature'] = long_df['temperature'].astype(int)
    
    # 准备训练特征和标签
    X_train = long_df[['wavenumber', 'temperature']]
    y_train = long_df['cross_section']
    
    # 处理测试集
    test_long_df = pd.melt(
        test_df,
        id_vars='wavenumber',
        value_vars=['t=1800k', 't=1900k', 't=2000k'],
        var_name='temperature',
        value_name='cross_section'
    )
    
    test_long_df['temperature'] = test_long_df['temperature'].str.replace('t=', '', regex=False).str.replace('k', '', regex=False).astype(int)
    
    X_test = test_long_df[['wavenumber', 'temperature']]
    y_test = test_long_df['cross_section']
    
    print(f"训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    
    # 建立模型
    degree = 6
    alpha = 1.0
    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        Lasso(alpha=alpha, max_iter=5000)
    )
    
    # 训练
    print("开始训练模型...")
    model.fit(X_train, y_train)
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 计算各种评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 计算MAPE，避免除0，加入一个小常数1e-8
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100  # 百分比形式
    
    # 计算预测误差的几何平均数，加入小常数避免log(0)
    geometric_mean_error = gmean(np.abs(y_test - y_pred) + 1e-8)
    
    # 打印评估指标
    print(f"\n{dataset_name} 评估结果:")
    print(f"Test Mean Squared Error (MSE): {mse:.4e}")
    print(f"Test Root Mean Squared Error (RMSE): {rmse:.4e}")
    print(f"Test Mean Absolute Error (MAE): {mae:.4e}")
    print(f"Test R-squared (R2): {r2:.4f}")
    print(f"Test Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Test Geometric Mean Error (GME): {geometric_mean_error:.4e}")
    
    return {
        'dataset': dataset_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'gme': geometric_mean_error,
        'y_test': y_test,
        'y_pred': y_pred
    }

# 主程序
if __name__ == "__main__":
    # 定义数据文件路径
    datasets = {
        'H2S v=1': 'h2s_v=1.xlsm',
        'H2S v=10': 'h2s_v=10.xlsm',
        'H2S v=20': 'h2s_v=20.xlsm'
    }
    
    results = {}
    processed_data = {}
    
    # 处理每个数据集
    for name, file_path in datasets.items():
        try:
            # 数据预处理
            train_df, test_df, stats_df = process_dataset(file_path, name)
            processed_data[name] = {
                'train': train_df,
                'test': test_df,
                'stats': stats_df
            }
            
            # 建模和评估
            result = train_and_evaluate(train_df, test_df, name)
            results[name] = result
            
        except Exception as e:
            print(f"处理 {name} 时出错: {str(e)}")
            continue
    
    # 创建对比表格
    print("\n" + "="*80)
    print("三个数据集性能对比")
    print("="*80)
    
    comparison_df = pd.DataFrame([
        {
            '数据集': result['dataset'],
            'MSE': f"{result['mse']:.4e}",
            'RMSE': f"{result['rmse']:.4e}",
            'MAE': f"{result['mae']:.4e}",
            'R²': f"{result['r2']:.4f}",
            'MAPE(%)': f"{result['mape']:.2f}",
            'GME': f"{result['gme']:.4e}"
        }
        for result in results.values()
    ])
    
    print(comparison_df.to_string(index=False))
    
    # 创建可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('H2S 三个振动态数据集建模性能对比', fontsize=16)
    
    # 1. 散点图对比
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green']
    for i, (name, result) in enumerate(results.items()):
        ax1.scatter(result['y_test'], result['y_pred'], 
                   alpha=0.6, s=10, color=colors[i], label=name)
    
    # 添加y=x线
    all_y_test = np.concatenate([result['y_test'] for result in results.values()])
    ax1.plot([all_y_test.min(), all_y_test.max()], 
             [all_y_test.min(), all_y_test.max()], 'k--', label='y=x')
    ax1.set_xlabel('True Cross Section')
    ax1.set_ylabel('Predicted Cross Section')
    ax1.set_title('True vs Predicted (所有数据集)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. R²对比
    ax2 = axes[0, 1]
    r2_values = [result['r2'] for result in results.values()]
    dataset_names = [result['dataset'] for result in results.values()]
    bars = ax2.bar(dataset_names, r2_values, color=colors)
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² 对比')
    ax2.set_ylim(0, 1)
    # 添加数值标签
    for bar, value in zip(bars, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. RMSE对比
    ax3 = axes[1, 0]
    rmse_values = [result['rmse'] for result in results.values()]
    bars = ax3.bar(dataset_names, rmse_values, color=colors)
    ax3.set_ylabel('RMSE')
    ax3.set_title('RMSE 对比')
    ax3.set_yscale('log')  # 使用对数刻度
    # 添加数值标签
    for bar, value in zip(bars, rmse_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{value:.2e}', ha='center', va='bottom', rotation=45)
    
    # 4. MAPE对比
    ax4 = axes[1, 1]
    mape_values = [result['mape'] for result in results.values()]
    bars = ax4.bar(dataset_names, mape_values, color=colors)
    ax4.set_ylabel('MAPE (%)')
    ax4.set_title('MAPE 对比')
    # 添加数值标签
    for bar, value in zip(bars, mape_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_values)*0.01,
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('h2s_three_datasets_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 单独为每个数据集创建散点图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('各数据集预测效果详细对比', fontsize=16)
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        ax.scatter(result['y_test'], result['y_pred'], alpha=0.6, s=15, color=colors[i])
        
        # 添加y=x线
        y_min, y_max = result['y_test'].min(), result['y_test'].max()
        ax.plot([y_min, y_max], [y_min, y_max], 'r--', alpha=0.8, label='y=x')
        
        ax.set_xlabel('True Cross Section')
        ax.set_ylabel('Predicted Cross Section')
        ax.set_title(f'{name}\nR²={result["r2"]:.3f}, RMSE={result["rmse"]:.2e}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('h2s_individual_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存对比结果到Excel
    with pd.ExcelWriter('h2s_comparison_results.xlsx') as writer:
        comparison_df.to_excel(writer, sheet_name='性能对比', index=False)
        
        # 保存每个数据集的统计信息
        for name, data in processed_data.items():
            sheet_name = name.replace('H2S ', '').replace('=', '')
            data['stats'].to_excel(writer, sheet_name=f'{sheet_name}_stats', index=False)
    
    print("\n分析完成！")
    print("生成的文件:")
    print("1. h2s_three_datasets_comparison.png - 综合性能对比图")
    print("2. h2s_individual_predictions.png - 各数据集详细预测图")
    print("3. h2s_comparison_results.xlsx - 详细对比结果")
    
    # 输出最佳模型
    best_r2 = max(results.values(), key=lambda x: x['r2'])
    best_rmse = min(results.values(), key=lambda x: x['rmse'])
    
    print(f"\n最佳R²: {best_r2['dataset']} (R² = {best_r2['r2']:.4f})")
    print(f"最低RMSE: {best_rmse['dataset']} (RMSE = {best_rmse['rmse']:.4e})")