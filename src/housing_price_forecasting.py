"""加州房价预测 —— 参考《Hands-On Machine Learning》(Aurélien Géron, O'Reilly) 第 2 章

原书示例代码：https://github.com/ageron/handson-ml2 （第 2 版，Apache License 2.0）
本文件在原书代码基础上改写，为适配 pandas 3 / scikit-learn 1.9 共改动 5 处，
均以 [修复N] 注释标注原因。
"""

import os
from pathlib import Path
import tarfile
import urllib.request
import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
from  sklearn.impute import SimpleImputer
from  sklearn.preprocessing import  OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import  mean_squared_error
from  sklearn.model_selection import  train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# 正确地址
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def split_train_test(data, test_ratio):
    # 1. 生成打乱后的索引（把行号随机洗牌）
    #permutation:不修改原数据，返回新数组，可以直接生成索引，适合数据集划分，即生成一个temp
    shuffled_indices = np.random.permutation(len(data))
    # 2. 算出测试集应该多少行（总条数 × 测试比例0.2）
    test_set_size = int(len(data) * test_ratio)
    # 3. 前20%索引 = 测试集索引
    test_indices = shuffled_indices[:test_set_size]
    # 4. 后80%索引 = 训练集索引
    train_indices = shuffled_indices[test_set_size:]#一切皆对象
    # 5. 按索引切分数据，返回训练集、测试集
    return data.iloc[train_indices], data.iloc[test_indices]

#1. 从网上获取数据
# fetch_housing_data()
#数据信息获取
# [修复1] "../data/..." 是按「当前工作目录」算的，从项目根目录运行（PyCharm 直接点运行）
#         就会找不到文件。改成从脚本自身位置往上找，任何目录启动都能读到
housing=pd.read_csv(Path(__file__).resolve().parents[1] / "data/housing/housing.csv")
#数据信息
print(housing.describe())
print('\n')
#前10行
print(housing.head(10))
print('\n')
df_bad=housing[housing["population"]<=10]
print(df_bad.sort_values(by="latitude",ascending=True))
print('\n')
#数据类型
print(housing.info())
print('\n')
#相关性矩阵
# [修复2] pandas 3 的 corr() 不再自动跳过文本列，必须显式 numeric_only
corr_matrix=housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))


#数据清洗
#一.数据处理
#1.
housing.dropna(subset=["total_bedrooms"])
print('\n')
print(housing.index)
#2.
median=housing["total_bedrooms"].median()
# [修复3] pandas 3 的 copy-on-write 下不允许链式 inplace，改为显式赋值
housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)
#3.利用machine learning中的simpleinpular
housing_num=housing.drop("ocean_proximity",axis=1)
Imputer=SimpleImputer(strategy="median")
Imputer.fit(housing_num)
#每一列的中位数
print(Imputer.statistics_)
X=Imputer.transform(housing_num)
#X为转换后的numpy数组
housing_tr=pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)

#二.文本处理和分类属性
print('\n')
housing_cat=housing["ocean_proximity"]
#独热向量【0，0，0，0，1】或【0，0，1，0，0】一1全0
# [修复4] sklearn 1.4 起 sparse 参数改名为 sparse_output（下面流水线里还有一处）
cat_encoder=OneHotEncoder(sparse_output=False)
#识别数字规律并自动填充
housing_cat_1hot=cat_encoder.fit_transform(housing_cat.values.reshape(-1,1))
print(housing_cat_1hot)

# #直方图
# housing.hist(bins=50,figsize=(20,15))#直方图
# plt.tight_layout()
# plt.show()


#自定义转换器
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


#转换流水线
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scaler',StandardScaler()),
])

# housing_num_tr=num_pipeline.fit_transform(housing_num)


housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value", axis=1)
num_attribs = list(housing.drop("ocean_proximity", axis=1).columns)       # 所有数值列的名字
cat_attribs=["ocean_proximity"]              # 文本列
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    # [修复4] 同上，sparse → sparse_output
    ("cat",OneHotEncoder(sparse_output=False,handle_unknown='ignore'),cat_attribs),
])

#           处理特征
housing_prepared=full_pipeline.fit_transform(housing)
print("housing_prepared shape:", housing_prepared.shape)

#拆分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(
    housing_prepared, housing_labels,
    test_size=0.2,
    random_state=42
)






# ========= 调试：加在 train_test_split 之后、训练之前 =========

# 1. 检查 housing_prepared 的统计
print("=== housing_prepared 诊断 ===")
print("shape:", housing_prepared.shape)
print("每列均值:", np.mean(housing_prepared, axis=0)[:5], "...")
print("每列标准差:", np.std(housing_prepared, axis=0)[:5], "...")
print("任意列标准差为0?", np.any(np.std(housing_prepared, axis=0) == 0))
print("最大值:", np.max(housing_prepared))
print("最小值:", np.min(housing_prepared))

# 2. 检查 train_test_split 之后的 X_train
print("\n=== X_train 诊断 ===")
print("shape:", X_train.shape)
print("每列均值:", np.mean(X_train, axis=0)[:5], "...")

# 3. 训练并检查系数
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print("\n=== 模型系数诊断 ===")
print("截距:", lin_reg.intercept_)
print("系数范围:", np.min(lin_reg.coef_), "~", np.max(lin_reg.coef_))
print("系数绝对值 > 1e6 的个数:", np.sum(np.abs(lin_reg.coef_) > 1e6))

# 4. 检查预测值
y_pred = lin_reg.predict(X_test)
print("\n=== 预测值诊断 ===")
print("预测范围:", np.min(y_pred), "~", np.max(y_pred))
print("真实范围:", np.min(y_test), "~", np.max(y_test))
print("前5个预测:", y_pred[:5])
print("前5个真实:", y_test.values[:5])
print('\n\n\n\n')






#       训练
lin_reg=LinearRegression()
lin_reg.fit(X_train, y_train)
# print("是否有 inf:", np.any(np.isinf(housing_prepared)))
# print("是否有 NaN:", np.any(np.isnan(housing_prepared)))
# print(housing[housing["households"] == 0].shape[0], "行的 households = 0")
# print(housing[housing["total_rooms"] == 0].shape[0], "行的 total_rooms = 0")
#       评估
y_pred=lin_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(f"线性回归 RMSE: {rmse:.0f}")
print(f"房价均值: {housing_labels.mean():.0f}")
print(f"RMSE / 均值: {rmse / housing_labels.mean() * 100:.1f}%")
# ============ 前5条直观对比 ============
print("\n前5条预测 vs 真实:")

#到这里前面其实都ok，就是后面交叉验证出问题，主要原因在于测试集被用于训练了？

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
for pred, real in zip(lin_reg.predict(some_data_prepared), some_labels):
    print(f"  预测: {pred:>10.0f} | 真实: {real:>10.0f} | 差: {abs(pred-real):>10.0f}")


#           交叉验证
# [修复5] 这份 CSV 按经纬度分块排列，cross_val_score 对回归默认 KFold(shuffle=False)，
#         不打乱就等于「用北加州训练、拿南加州测试」，每折都在跨区域外推，
#         RMSE 会虚高 25%、标准差高达 ±1.4 万。必须 shuffle。
cv10 = KFold(n_splits=10, shuffle=True, random_state=42)

final_pipeline = Pipeline([
    ('prep', full_pipeline),       # 预处理
    ('ridge', Ridge(alpha=1.0)),    # 模型
])
lin_scores = cross_val_score(final_pipeline, housing, housing_labels,
                             scoring="neg_mean_squared_error", cv=cv10)
lin_rmse = np.sqrt(-lin_scores)
print(f"\n线性回归 10折交叉验证 RMSE: {lin_rmse.mean():.0f} ± {lin_rmse.std():.0f}")



#           对比模型
tree_scores=cross_val_score(DecisionTreeRegressor(random_state=42),
                              housing_prepared, housing_labels,
                              scoring="neg_mean_squared_error", cv=cv10)
tree_rmse = np.sqrt(-tree_scores)
print(f"决策树   10折交叉验证 RMSE: {tree_rmse.mean():.0f} ± {tree_rmse.std():.0f}")


forest_scores = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42),
                                housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=cv10)
forest_rmse = np.sqrt(-forest_scores)
print(f"随机森林 10折交叉验证 RMSE: {forest_rmse.mean():.0f} ± {forest_rmse.std():.0f}")
print("\n===== 模型对比 =====")
print(f"线性回归: {lin_rmse.mean():.0f}")
print(f"决策树:   {tree_rmse.mean():.0f}")
print(f"随机森林: {forest_rmse.mean():.0f}")
