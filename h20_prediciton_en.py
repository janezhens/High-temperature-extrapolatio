import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import joblib
import warnings
warnings.filterwarnings('ignore')

# 分子参数
MOLECULE_PARAMS = {
    'H2S': {
        'InChI': '1S/H2S/h1H2',
        'molecule_weight': 34.081,
        'molecule_dipole': 0.98,
        'B0a': 10.34657,
        'B0b': 9.03572,
        'B0c': 4.7268
    },
    'PH3': {
        'InChI': '1S/H3P/h1H3',
        'molecule_weight': 33.99758,
        'molecule_dipole': 0.58,
        'B0a': 4.45236,
        'B0b': 4.45236,
        'B0c': 3.93
    },
    'SO2': {
        'InChI': '1S/O2S/c1-3-2',
        'molecule_weight': 64.064,
        'molecule_dipole': 1.6331,
        'B0a': 2.0273542038205643,
        'B0b': 0.3441739017997577,
        'B0c': 0.2935264969207464
    },
    'SiO2': {
        'InChI': '1S/O2Si/c1-3-2',
        'molecule_weight': 60.0843,
        'molecule_dipole': 0.0,
        'B0a': 0.7088,
        'B0b': 0.7088,
        'B0c': 0.3544
    },
    'H2O': {
        'InChI': '1S/H2O/h1H2',
        'molecule_weight': 18.0153,
        'molecule_dipole': 1.857,
        'B0a': 27.880631339965202,
        'B0b': 14.521769590347734,
        'B0c': 9.277708380509026
    },
    'HCN': { 
        'InChI': '1S/CHN/c1-2/h1H', 
        'molecule_weight': 27.0253, 
        'molecule_dipole': 2.98, 
        'B0a': 100000.0,
        'B0b': 1.47822, 
        'B0c': 1.47822 
    }
}

def preprocess_molecule_data(df, params):
    """预处理分子数据"""
    df_long = pd.melt(
        df,
        id_vars='wavenumber',
        var_name='temperature',
        value_name='cross_section'
    )
    df_long['temperature'] = df_long['temperature'].str.replace('t=', '', regex=False).str.replace('k', '', regex=False).astype(int)
    for key, value in params.items():
        df_long[key] = value
    return df_long

def feature_engineering_consistent(df, all_inchi_values=None, all_temp_ranges=None):
    """一致性特征工程，确保训练和预测时特征完全相同"""
    df_eng = df.copy()
    numeric_cols = ['wavenumber', 'temperature', 'cross_section', 'molecule_weight', 'molecule_dipole', 'B0a', 'B0b', 'B0c']
    for col in numeric_cols:
        df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce')
    
    # 基础特征变换
    df_eng['wavenumber_log'] = np.log1p(df_eng['wavenumber'])
    df_eng['temperature_log'] = np.log(df_eng['temperature'])
    df_eng['wn_temp_interaction'] = df_eng['wavenumber'] * df_eng['temperature']
    df_eng['wn_temp_ratio'] = df_eng['wavenumber'] / df_eng['temperature']
    df_eng['dipole_weight_ratio'] = df_eng['molecule_dipole'] / df_eng['molecule_weight']
    df_eng['rotational_energy'] = df_eng['B0a'] * df_eng['B0b'] * df_eng['B0c']
    df_eng['rotational_asymmetry'] = (2*df_eng['B0b'] - df_eng['B0a'] - df_eng['B0c']) / (df_eng['B0a'] - df_eng['B0c'] + 1e-10)
    
    # 高级特征工程
    df_eng['temp_wavenumber_squared'] = df_eng['temperature'] * (df_eng['wavenumber']**2)
    df_eng['dipole_energy_interaction'] = df_eng['molecule_dipole'] * df_eng['rotational_energy']
    df_eng['temp_dipole_interaction'] = df_eng['temperature'] * df_eng['molecule_dipole']
    df_eng['wavenumber_dipole_interaction'] = df_eng['wavenumber'] * df_eng['molecule_dipole']
    
    # 物理启发的特征
    df_eng['boltzmann_factor'] = np.exp(-0.695 * df_eng['wavenumber'] / df_eng['temperature'])
    df_eng['quantum_factor'] = 1 - np.exp(-1.439 * df_eng['wavenumber'] / df_eng['temperature'])
    df_eng['rotational_partition'] = np.sqrt(df_eng['temperature']**3 / (df_eng['B0a'] * df_eng['B0b'] * df_eng['B0c']))
    
    # 一致性分类特征处理
    if all_inchi_values is None:
        all_inchi_values = list(set([params['InChI'] for params in MOLECULE_PARAMS.values()]))
    
    # 为所有可能的InChI值创建虚拟变量
    for inchi in all_inchi_values:
        df_eng[f'molecule_{inchi}'] = (df_eng['InChI'] == inchi).astype(int)
    
    # 温度范围分类
    df_eng['temp_range'] = pd.cut(df_eng['temperature'], bins=[0, 300, 800, 1500, 3000], labels=['low', 'medium', 'high', 'very_high'])
    
    if all_temp_ranges is None:
        all_temp_ranges = ['low', 'medium', 'high', 'very_high']
    
    # 为所有可能的温度范围创建虚拟变量
    for temp_range in all_temp_ranges:
        df_eng[f'temp_{temp_range}'] = (df_eng['temp_range'] == temp_range).astype(int)
    
    # 删除原始分类列
    df_eng = df_eng.drop(['InChI', 'temp_range'], axis=1)
    
    # 目标变量处理
    epsilon = 1e-50
    df_eng['cross_section_safe'] = df_eng['cross_section'] + epsilon
    df_eng['log_cross_section'] = np.log10(df_eng['cross_section_safe'])
    
    return df_eng

def prepare_features_consistent():
    """准备一致的特征列表"""
    basic_features = ['wavenumber', 'temperature', 'molecule_weight', 'molecule_dipole', 'B0a', 'B0b', 'B0c']
    
    engineered_features = [
        'wavenumber_log', 'temperature_log', 'wn_temp_interaction', 'wn_temp_ratio',
        'dipole_weight_ratio', 'rotational_energy', 'rotational_asymmetry',
        'temp_wavenumber_squared', 'dipole_energy_interaction', 'temp_dipole_interaction',
        'wavenumber_dipole_interaction', 'boltzmann_factor', 'quantum_factor', 'rotational_partition'
    ]
    
    # 所有可能的分子特征
    all_inchi_values = list(set([params['InChI'] for params in MOLECULE_PARAMS.values()]))
    molecule_cols = [f'molecule_{inchi}' for inchi in all_inchi_values]
    
    # 所有可能的温度范围特征
    temp_cols = ['temp_low', 'temp_medium', 'temp_high', 'temp_very_high']
    
    all_features = basic_features + engineered_features + molecule_cols + temp_cols
    
    return all_features, all_inchi_values, ['low', 'medium', 'high', 'very_high']

def select_best_features(X, y, n_features=20):
    """使用随机森林进行特征重要性排序和选择"""
    print("Performing feature selection...")
    selector = SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=42), 
        max_features=n_features,
        threshold=-np.inf
    )
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Selected {len(selected_features)} features: {', '.join(selected_features[:5])}...")
    return selected_features

def build_advanced_model():
    """构建高级集成模型"""
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    nn = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    ensemble = VotingRegressor([
        ('rf', rf),
        ('gb', gb),
        ('nn', nn)
    ])
    
    model_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', ensemble)
    ])
    
    return model_pipeline

def main():
    start_time = time.time()
    print("\n=== Fixed H2O Cross Section Prediction Model ===\n")
    
    # 准备一致的特征列表
    all_features, all_inchi_values, all_temp_ranges = prepare_features_consistent()
    
    # 加载训练数据
    print("Loading training data...")
    df_h2s = pd.read_excel('h2s_v=20.xlsm')
    df_ph3 = pd.read_excel('ph3_v=20.xlsm')
    df_so2 = pd.read_excel('so2_v=20.xlsm')
    df_sio2 = pd.read_excel('sio2_v=20.xlsm')
    df_hcn = pd.read_excel('HCN_output1.xlsm')
    
    # 数据预处理
    df_h2s_long = preprocess_molecule_data(df_h2s, MOLECULE_PARAMS['H2S'])
    df_ph3_long = preprocess_molecule_data(df_ph3, MOLECULE_PARAMS['PH3'])
    df_so2_long = preprocess_molecule_data(df_so2, MOLECULE_PARAMS['SO2'])
    df_sio2_long = preprocess_molecule_data(df_sio2, MOLECULE_PARAMS['SiO2'])
    df_hcn_long = preprocess_molecule_data(df_hcn, MOLECULE_PARAMS['HCN'])
    
    # 合并训练数据
    df_train = pd.concat([df_h2s_long, df_ph3_long, df_so2_long, df_sio2_long, df_hcn_long], ignore_index=True)
    print(f"Training data shape: {df_train.shape}")
    
    # 加载H2O数据
    print("Loading H2O data for prediction...")
    df_h2o = pd.read_excel('h2o_output1.xlsm')
    df_h2o_long = preprocess_molecule_data(df_h2o, MOLECULE_PARAMS['H2O'])
    print(f"H2O data shape: {df_h2o_long.shape}")
    
    # 一致性特征工程
    print("Performing consistent feature engineering...")
    df_train_eng = feature_engineering_consistent(df_train, all_inchi_values, all_temp_ranges)
    df_h2o_eng = feature_engineering_consistent(df_h2o_long, all_inchi_values, all_temp_ranges)
    
    # 确保所有特征都存在
    for feature in all_features:
        if feature not in df_train_eng.columns:
            df_train_eng[feature] = 0
        if feature not in df_h2o_eng.columns:
            df_h2o_eng[feature] = 0
    
    # 准备训练数据
    X_train = df_train_eng[all_features]
    y_train = df_train_eng['log_cross_section']
    mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]
    print(f"Clean training data shape: {X_train_clean.shape}")
    
    # 特征选择
    selected_features = select_best_features(X_train_clean, y_train_clean)
    X_train_selected = X_train_clean[selected_features]
    
    # 划分训练集和验证集
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_selected, y_train_clean, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train_final.shape}, Validation set: {X_val.shape}")
    
    # 构建和训练模型
    print("\nBuilding and training model...")
    model = build_advanced_model()
    
    # 交叉验证评估
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_final, y_train_final, cv=cv, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R² score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    
    # 在完整训练集上训练最终模型
    print("Training final model...")
    model.fit(X_train_selected, y_train_clean)
    
    # 预测H2O
    print("\nPredicting H2O cross section...")
    X_h2o = df_h2o_eng[selected_features].fillna(0)
    
    # 确保特征顺序完全一致
    X_h2o = X_h2o[selected_features]
    
    print(f"H2O prediction data shape: {X_h2o.shape}")
    print(f"Selected features: {list(selected_features)}")
    
    y_pred_log = model.predict(X_h2o)
    y_pred = 10**y_pred_log
    
    # 保存结果
    results_df = pd.DataFrame({
        'wavenumber': df_h2o_long['wavenumber'],
        'temperature': df_h2o_long['temperature'],
        'actual_cross_section': df_h2o_long['cross_section'],
        'predicted_cross_section': y_pred,
        'log_actual': np.log10(df_h2o_long['cross_section'] + 1e-50),
        'log_predicted': y_pred_log
    })
    
    results_df.to_csv('H2O_cross_section_fixed_predictions.csv', index=False)
    print("Prediction results saved to 'H2O_cross_section_fixed_predictions.csv'.")
    
    # 评估
    print("\nFinal evaluation on H2O:")
    mask_eval = (results_df['actual_cross_section'] > 0) & (results_df['predicted_cross_section'] > 0)
    df_eval = results_df[mask_eval]
    
    if len(df_eval) > 0:
        mse = mean_squared_error(df_eval['log_actual'], df_eval['log_predicted'])
        rmse = np.sqrt(mse)
        r2 = r2_score(df_eval['log_actual'], df_eval['log_predicted'])
        mae = mean_absolute_error(df_eval['log_actual'], df_eval['log_predicted'])
        rel_error = np.mean(np.abs(df_eval['actual_cross_section'] - df_eval['predicted_cross_section']) / df_eval['actual_cross_section'])
        
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Mean relative error: {rel_error:.6f}")
    else:
        print("No valid data for evaluation.")
    
    # 保存模型和特征信息
    model_info = {
        'model': model,
        'selected_features': list(selected_features),
        'all_inchi_values': all_inchi_values,
        'all_temp_ranges': all_temp_ranges
    }
    joblib.dump(model_info, 'h2o_fixed_prediction_model.pkl')
    print("Model and feature info saved as 'h2o_fixed_prediction_model.pkl'.")
    
    # 计算运行时间
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()