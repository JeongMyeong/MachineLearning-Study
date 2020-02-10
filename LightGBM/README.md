### 설치

```
pip install lightgbm
```

### Base Code
```
import lightgbm as lgb
params = {
    'bagging_fraction':0.9,
    'feature_fraction':0.9,
    'num_leaves': 12, 
    'max_depth': 2,
    'num_class':4,
    'objective': 'multiclass',
    'metric':'multi_logloss',
    'num_threads': 14,
    'learning_rate': 0.1,
    'is_unbalance': True,
    'lambda_l1':2,
    'lambda_l2':3,
}

train_set = lgb.Dataset(X_train, label=y_train, params=data_params)              # lgb에 맞는 Dataset 형태로 변환 해주어야함.
# valid_set = lgb.Dataset(X_valid, label=y_valid, params=data_params)
num_round=100

lgb_model = lgb.train(params, train_set, num_round, early_stopping_rounds=50, 
                    valid_sets=[train_set, valid_set],
                    verbose_eval=30,
                  keep_training_booster=True,                                  # 나중에 이어서 모델을 학습하기 위한 파라미터
                  # model_init = model                                          # 이어서 모델을 
                   )

```
