"""加州房价预测 —— 读《Hands-On Machine Learning》第 2 章，跟着走一遍完整流程。

数据集：1990 年美国加州人口普查的街区房价（20640 行 × 10 列）。
目标：用其余 9 列预测 median_house_value（该街区房价中位数，单位美元）。

文件结构
    0  环境和路径
    1  读数据、看数据
    2  数据清洗（缺失值）
    3  特征工程（比值特征）
    4  预处理流水线（缺失值 → 比值特征 → 标准化 → 独热）
    5  切分训练/测试集
    6  线性回归：训练、预测、评估
    7  交叉验证：横向比较三种模型
    8  画图

运行
    cd ~/ml-practice && uv run python src/housing_price_forecasting.py

原始数据
    https://github.com/ageron/handson-ml2 （Apache License 2.0）
    下载地址见本节末尾的 fetch_housing_data()，已下载的可直接跳过。
"""

# ============================================================ 0. 环境和路径
import os
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 打印 DataFrame 时不要省略列、不要换行，否则 10 列会显示成两截
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# 数据文件的绝对路径，从「本文件所在目录」往上退一级再进 data/
# 这样不管从项目根目录、src/ 里、还是用 PyCharm 直接点运行，都能找到文件
DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "housing" / "housing.csv"

# 图保存到这个目录
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


# ------------------------------------------------------------ 下载数据（可选）
# 第一次用才需要，数据已经在本地的直接跳过
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=DATA_CSV.parent):
    """下载官方 tgz 并解压到 data/housing/ 下，得到 housing.csv。"""
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as f:
        f.extractall(path=housing_path)


# 数据已存在，不用每次下载
# fetch_housing_data()


# ============================================================ 1. 读数据、看数据
housing = pd.read_csv(DATA_CSV)

# describe 输出 8 个统计量：count / mean / std / min / 25% / 50% / 75% / max
# 重点看 max 那一行：房价最大值 500001 是个可疑的整数，后面会看到它是人工截顶的结果
print("=== 数据概览 ===")
print(housing.describe())
print("\n")

# 前 10 行，肉眼确认列名和量级
print(housing.head(10))
print("\n")

# 人口极少的街区（<=10 人）通常是空房或度假屋，价格统计不可靠
# 这里只是列出来看看，不删
print(housing[housing["population"] <= 10].sort_values(by="latitude"))
print("\n")

# info() 看每列的数据类型和是否有缺失值
# 会看到 total_bedrooms 只有 20433 个非空值，缺 207 条
print(housing.info())
print("\n")

# 相关性：每列与房价的线性相关程度，取值 -1 到 1
# numeric_only=True 必须加：pandas 3 的 corr() 遇到文本列不再自动跳过，会直接报错
print(housing.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False))


# ============================================================ 2. 数据清洗
# total_bedrooms 缺 207 条，用中位数填。
# 为什么用中位数不用均值：中位数对极端值不敏感，而这列有少数街区卧室数极高。
print("\n=== 缺失值处理 ===")
median = housing["total_bedrooms"].median()
print(f"total_bedrooms 缺失 {housing['total_bedrooms'].isna().sum()} 条，用中位数 {median} 填充")

# 显式赋值，不要用 inplace=True：
# pandas 3 默认开启 copy-on-write，对切片做 inplace 修改会抛 ChainedAssignmentError
housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)


# ============================================================ 3. 特征工程
# 原始列是「这个街区的总量」，受街区大小影响很大。
# 换成比值之后才反映密度和结构，和房价的关系更直接：
#     rooms_per_household        每户几间房  → 房子大不大
#     population_per_household   每户几口人  → 挤不挤
#     bedrooms_per_room          几成是卧室  → 户型结构
# 注意：这个转换器接收的是 numpy 数组，只能用列下标取列，不能用列名。
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """在原有特征后面拼接三个比值特征。

    继承 BaseEstimator 和 TransformerMixin 有实际作用：
      - TransformerMixin 自动提供 fit_transform()，不用自己写
      - BaseEstimator 提供 get_params()/set_params()，Pipeline 和网格搜索靠它传递参数
    """

    def __init__(self, add_bedrooms_per_room=True):
        # 参数只允许在 __init__ 里赋值并存成同名属性，sklearn 靠属性名反查参数
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        # 比值特征不需要从数据里学习任何东西，所以 fit 什么都不做
        # 但必须返回 self，否则 Pipeline 拿不到可继续调用的对象
        return self

    def transform(self, X):
        # 列下标由下面的 _COL_IX 提供，顺序与 housing 的列一致
        rooms_per_household = X[:, _COL_IX["total_rooms"]] / X[:, _COL_IX["households"]]
        population_per_household = X[:, _COL_IX["population"]] / X[:, _COL_IX["households"]]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, _COL_IX["total_bedrooms"]] / X[:, _COL_IX["total_rooms"]]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        return np.c_[X, rooms_per_household, population_per_household]


# 用列名反查下标，而不是写死 3/4/5/6。
# 写死数字的隐患：只要列顺序变了（比如调整透 drop 的顺序），下标全错但不报错，
# 只是悄悄算出三个没意义的特征。
_COL_IX = {name: i for i, name in enumerate(housing.columns)}

# 单独试一下这个转换器，看看比值特征长什么样
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
print(f"\n加上比值特征后：{housing.values.shape[1]} 列 → {housing_extra_attribs.shape[1]} 列")


# ============================================================ 4. 预处理流水线
# 先取出特征和目标，后面所有地方都只对「特征」做处理。
# 如果把带 median_house_value 的整张表交给 ColumnTransformer，
# num_attribs 里就会混进目标列，导致目标被当成特征喂给模型。
housing_labels = housing["median_house_value"].copy()          # 要预测的目标
housing_features = housing.drop("median_house_value", axis=1)  # 输入特征

# 数值列和文本列要分开处理，最后拼起来。ColumnTransformer 负责按列名分发。
num_attribs = [c for c in housing_features.columns if c != "ocean_proximity"]
cat_attribs = ["ocean_proximity"]

# 数值流水线，按顺序执行三步：
#   1) SimpleImputer   缺失值填中位数（流水线里再做一次，保证训练集之外的缺失也能处理）
#   2) 比值特征        加三个派生列
#   3) StandardScaler  标准化到均值 0、方差 1
# 标准化对线性模型和 Ridge 是必需的（否则系数尺度不可比、正则项对大尺度特征惩罚更重），
# 对决策树没用但无害，统一放进来省事。
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler()),
])

# 文本列 ocean_proximity 只有 5 个取值（<1H OCEAN / INLAND / ISLAND / NEAR BAY / NEAR OCEAN），
# 用独热编码转成 5 列 0/1。
#   sparse_output=False 必须写：sklearn 1.4 起 dense 参数改名为 sparse_output,
#                       用旧名字 sparse=False 会直接 TypeError
#   handle_unknown="ignore" 防止上线后遇到训练时没见过的类别就崩，遇到时整行为 0
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_attribs),
])


# ============================================================ 5. 切分训练/测试集
# 先切分，再 fit 预处理。顺序不能反，原因见下面第 7 节的说明。
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    housing_features,
    housing_labels,
    test_size=0.2,
    random_state=42,      # 固定随机种子，保证每次切出同一份测试集，结果可复现
)

# fit_transform 只用在训练集上：均值、标准差、中位数、类别表都只从训练集统计
X_train = full_pipeline.fit_transform(X_train_raw)
# 测试集只 transform：用训练集学到的统计量去变换，自己一点信息都不提供
X_test = full_pipeline.transform(X_test_raw)

print(f"\n训练集 {X_train.shape}，测试集 {X_test.shape}")


# ============================================================ 6. 线性回归
def rmse(y_true, y_pred):
    """均方根误差：先算均方误差再开根号，单位回到美元。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
base_rmse = rmse(y_test, y_pred)

print("\n=== 线性回归（测试集） ===")
print(f"RMSE     : {base_rmse:,.0f}")
print(f"房价均值 : {housing_labels.mean():,.0f}")
print(f"误差占比 : {base_rmse / housing_labels.mean() * 100:.1f}%")

# 拿前 5 条对比预测值和真实值，直观感受误差量级
print("\n前 5 条预测 vs 真实：")
for pred, real in zip(lin_reg.predict(X_test[:5]), y_test.iloc[:5]):
    print(f"  预测 {pred:>10,.0f} | 真实 {real:>10,.0f} | 差 {abs(pred - real):>9,.0f}")


# ============================================================ 7. 交叉验证
# 第 6 节只用了「一组」训练/测试切分，RMSE 会随切法波动。
# 交叉验证把数据切成 K 份，轮流拿其中 1 份当验证集，得到 K 个 RMSE 取平均，
# 结果更稳，也能给出 ±标准差 表示波动范围。
#
# 为什么必须把模型和预处理一起包进 Pipeline 再交给 cross_val_score：
#   cross_val_score 每一折都会重新调用 pipe.fit()。预处理放在 Pipeline 内部，
#   每一折就只用这一折的训练部分去 fit 标准化和填充；
#   如果先把整个数据集 fit_transform 好再传进去，标准化用到了全部数据的均值和标准差，
#   验证折的信息提前泄漏进训练过程，分数会偏乐观。
def cv_rmse(model, name):
    """把预处理和模型打包成一条流水线，跑 10 折交叉验证，返回平均 RMSE。"""
    pipe = Pipeline([
        ("prep", full_pipeline),
        ("model", model),
    ])
    # shuffle=True 一定要加：回归任务的 cross_val_score 默认用 KFold(shuffle=False)，
    # 也就是按原始行顺序切成 K 个连续块。这份 CSV 的原始排序带有地域聚集性，
    # 连续块之间彼此差异较大，不打乱会让每一折的验证集分布偏离整体分布，方差偏大。
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipe, housing_features, housing_labels,
        scoring="neg_mean_squared_error",   # 回归用负 MSE，因为 sklearn 约定分数越大越好
        cv=cv,
        n_jobs=-1,                          # 用满所有 CPU 核
    )
    errs = np.sqrt(-scores)                 # 取负号还原，再开根号回到美元
    print(f"{name:<10} RMSE {errs.mean():>9,.0f}  ± {errs.std():>6,.0f}"
          f"    单折 {errs.min() / 1000:.0f}k ~ {errs.max() / 1000:.0f}k")
    return errs.mean()


print("\n=== 10 折交叉验证 ===")
results = {
    # 用 Ridge 而不是 LinearRegression：Ridge 带 L2 正则，在特征共线性时更稳，
    # alpha 是正则强度，越大约束越强
    "线性回归": cv_rmse(Ridge(alpha=1.0), "线性回归"),
    "决策树": cv_rmse(DecisionTreeRegressor(random_state=42), "决策树"),
    "随机森林": cv_rmse(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "随机森林"
    ),
}

print("\n===== 模型对比 =====")
for name, score in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"  {name:<10} {score:>10,.0f}")


# ============================================================ 8. 画图
# 左图：预测 vs 真实。点落在红色对角线上表示预测准确。
# 右图：残差分布。残差 = 真实 - 预测，理想情况是以 0 为中心、左右对称的钟形。
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y_test, y_pred, s=6, alpha=0.4)
lims = [housing_labels.min(), housing_labels.max()]
axes[0].plot(lims, lims, "r--", lw=1)
axes[0].set_xlabel("真实房价")
axes[0].set_ylabel("预测房价")
axes[0].set_title("线性回归：预测 vs 真实")

residuals = y_test - y_pred
axes[1].hist(residuals, bins=60, color="steelblue")
axes[1].axvline(0, color="crimson", ls="--", lw=1)
axes[1].set_xlabel("残差（真实 - 预测）")
axes[1].set_title(f"残差分布（RMSE={base_rmse:,.0f}）")

plt.tight_layout()
REPORT_DIR.mkdir(exist_ok=True)
out = REPORT_DIR / "housing_prediction.png"
plt.savefig(out, dpi=120)
print(f"\n图已保存：{out}")
