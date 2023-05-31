print('import library...')
import warnings

warnings.filterwarnings("ignore")


import os
from torch.utils import data
from dataset import*
from config import settings
from train_test import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import time
def format_time(seconds):
    minutes = seconds // 60
    hours = minutes // 60

    if hours > 0:
        return f"{hours} hr {minutes % 60} min {seconds % 60} sec"
    elif minutes > 0:
        return f"{minutes} min {seconds % 60} sec"
    else:
        return f"{seconds} sec"
transformers = {
        'year': None, # Normalize
        'weekday': None,
        'time': None, # Standardlize
        'weather': None
    }

print('loadind data...')
train_df, transformers = load_data('train', total_sample=4000000, random_sample=settings.totalN, scaling_transformers=transformers)
print(f'\nAfter feature engineering: {train_df.columns},\n total len: {len(train_df.columns)}\n')
test_df, transformers = load_data('test', random_sample=settings.totalN, scaling_transformers=transformers)
train_data_df = train_df.copy()
test_data_df = test_df.copy()
train_data_df.drop('fare_amount', axis=1, inplace=True)
train_data_df.drop('total_fixed_fees', axis=1, inplace=True)
test_data_df.drop('total_fixed_fees', axis=1, inplace=True)
test_data_df.drop('key', axis=1, inplace=True)


X_train = train_data_df.drop('fare_amount', axis=1)
y_train = train_data_df['fare_amount']

# PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=4)
# principalComponents_train = pca.fit_transform(X_train)
# principalComponents_test = pca.transform(test_data_df)
# principalDf_train = pd.DataFrame(data = principalComponents_train, columns = ['principal component ' + str(i) for i in range(1, 5)], index=X_train.index)
# principalDf_test = pd.DataFrame(data = principalComponents_test, columns = ['principal component ' + str(i) for i in range(1, 5)], index=test_data_df.index)
# X_train = pd.concat([X_train, principalDf_train], axis = 1)
# test_data_df = pd.concat([test_data_df, principalDf_test], axis = 1)

print('\ncreating XGBRegressor model...')

# 建立XGBoost回歸模型
os.environ['XGB_USE_CUDA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# BayesSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# 定義 XGBoost Regressor
xgb_reg = XGBRegressor()

# 定義超參數搜索範圍
params = {
    'max_depth': Integer(3, 15),
    'n_estimators': Integer(1000, 5000),
    'learning_rate': Real(0.001, 1.0, 'log-uniform'),
    'subsample': Real(0.5, 1.0, 'uniform'),
    'booster': ['gbtree', 'dart'],
    'colsample_bytree': Real(0.5, 1.0, 'uniform'),
    'alpha': Real(0.1, 1.0, 'uniform'),
}

# 初始化 BayesSearchCV
bayes_search = BayesSearchCV(
    xgb_reg,
    params,
    cv=5,
    n_iter=50,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# 擬合數據
bayes_search.fit(X_train, y_train)

# 找到最佳參數
best_params = bayes_search.best_params_
print(f"最佳參數: {best_params}")

# train model
model = XGBRegressor(params, tree_method= 'gpu_hist', gpu_id=0, random_state=42)   
print(model.get_params())

# 訓練模型
print('training model...')
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'total training time: {format_time(int(elapsed_time))}')
print('predict result...')
# 用訓練好的模型來預測測試資料集
predictions = model.predict(test_data_df)
# 將預測結果轉換為DataFrame
total_fixed_fees = test_df['total_fixed_fees'].values
predictions_df = pd.DataFrame(predictions+total_fixed_fees, columns=['fare_amount'])


# 從測試數據集中提取'key'欄位
keys_df = test_df['key']

# 將'key'與預測結果組合成一個新的DataFrame
result_df = pd.concat([keys_df.reset_index(drop=True), predictions_df], axis=1)
result_df.to_csv('result.csv', index=False)