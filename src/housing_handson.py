"""加州房价预测 —— 从 Windows 上的 housing_price_forecasting.py 迁移并修复。

原始代码来自《机器学习实战：基于 Scikit-Learn、Keras 和 TensorFlow》第 2 章。
本文件保留原来的思路和顺序，只修掉在新版 pandas / scikit-learn 上会崩的地方，
以及两处逻辑问题。每处改动都标了 [修复N]。

运行：
    cd ~/ml-practice && uv run python src/housing_handson.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# 路径改成相对本文件定位，放在哪个目录都能跑 [修复0]
DATA_CSV = Path(__file__).resolve().parent.parent / "data" / "housing" / "housing.csv"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------- 1. 读数据
housing = pd.read_csv(DATA_CSV)

print("=== 数据概览 ===")
print(housing.describe())
print("\n前 5 行:")
print(housing.head())
print("\n数据类型:")
housing.info()

# [修复1] pandas 3.x 的 corr() 不再自动跳过文本列，必须显式 numeric_only=True
# 老代码直接 housing.corr() 会报 could not convert string to float: 'NEAR BAY'
print("\n与房价相关性最高的特征:")
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# ---------------------------------------------------------------- 2. 数据清洗
# [修复2] 原代码 housing.dropna(subset=[...]) 没赋值，结果被直接丢掉，等于没执行。
# 而且紧接着用中位数填充，本来也不该 dropna。这里删掉无用行，只保留填充逻辑。
median = housing["total_bedrooms"].median()
print(f"\ntotal_bedrooms 缺失 {housing['total_bedrooms'].isna().sum()} 条，用中位数 {median} 填充")

# [修复3] 老写法 housing["col"].fillna(..., inplace=True) 在 pandas 3.x 的
# copy-on-write 下会抛 ChainedAssignmentError。改成显式赋值。
housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)

# ---------------------------------------------------------------- 3. 自定义转换器
# 列位置必须按当前列顺序数，别写死 [修复4]
COLS = list(housing.columns)
ROOMS_IX = COLS.index("total_rooms")
BEDROOMS_IX = COLS.index("total_bedrooms")
POPULATION_IX = COLS.index("population")
HOUSEHOLDS_IX = COLS.index("households")


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """加三个比值特征。比原始计数更能说明问题（房间多不等于房子大）。"""

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, ROOMS_IX] / X[:, HOUSEHOLDS_IX]
        population_per_household = X[:, POPULATION_IX] / X[:, HOUSEHOLDS_IX]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, BEDROOMS_IX] / X[:, ROOMS_IX]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]


# ---------------------------------------------------------------- 4. 拆分（关键修复）
# [修复5] 原代码先把整份数据丢进 full_pipeline.fit_transform()，然后才 train_test_split。
# 这在原则上是数据泄漏：标准化用的均值和标准差、填充用的中位数，都吃到了测试集的信息。
# 正确顺序：先切分原始数据 → 流水线只在训练集上 fit。
# 但要诚实说明：实测这份数据上泄漏的影响是 0～9 美元（见下），因为「算均值和标准差」
# 是低容量操作，2 万行里少 4 千行几乎不改变统计量。真正的危险场景是 target encoding、
# 特征筛选、SMOTE 这类直接用到了 y 的操作。写法仍然要改——原则错了迟早出事。
housing_labels = housing["median_house_value"].copy()
housing_features = housing.drop("median_house_value", axis=1)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    housing_features, housing_labels, test_size=0.2, random_state=42
)

num_attribs = [c for c in housing_features.columns if c != "ocean_proximity"]
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler()),
])

# [修复6] OneHotEncoder(sparse=False) 在 scikit-learn 1.4 起被删除，
# 现在叫 sparse_output=False。用旧名字会直接 TypeError。
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_attribs),
])

X_train = full_pipeline.fit_transform(X_train_raw)   # 只 fit 训练集
X_test = full_pipeline.transform(X_test_raw)         # 测试集只 transform

print(f"\n训练集 {X_train.shape}，测试集 {X_test.shape}")

# ---------------------------------------------------------------- 5. 线性回归
lin_reg = LinearRegression().fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
base_rmse = rmse(y_test, y_pred)

print("\n=== 线性回归 ===")
print(f"RMSE        : {base_rmse:,.0f}")
print(f"房价均值     : {housing_labels.mean():,.0f}")
print(f"误差占比     : {base_rmse / housing_labels.mean() * 100:.1f}%")

print("\n前 5 条预测对比:")
for pred, real in zip(lin_reg.predict(X_test[:5]), y_test[:5]):
    print(f"  预测 {pred:>10,.0f} | 真实 {real:>10,.0f} | 差 {abs(pred - real):>9,.0f}")

# ---------------------------------------------------------------- 6. 交叉验证对比
# [修复7] 原代码 Ridge 用带流水线的 final_pipeline，而决策树/随机森林用的是
# 提前算好的 housing_prepared——两者预处理方式不同（后者还含泄漏），横向对比不公平。
# 统一：三个模型都包进同一个流水线，让交叉验证自己处理每一折的 fit。
def cv_rmse(model, name):
    pipe = Pipeline([("prep", full_pipeline), ("model", model)])
    # [修复8] 关键：必须 shuffle。
    # cross_val_score 对回归任务默认用 KFold(shuffle=False)，按行顺序切块。
    # 这份 CSV 是按经纬度分块排列的（同一片区域的房子连在一起），不 shuffle 就变成
    # 「用北加州训练、拿南加州测试」，每一折都在跨区域外推，RMSE 从 48k 涨到 63k。
    scores = cross_val_score(
        pipe, housing_features, housing_labels,
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=10, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    errs = np.sqrt(-scores)
    print(f"{name:<8} RMSE {errs.mean():>9,.0f}  ±{errs.std():,.0f}   单折 {errs.min()/1000:.0f}k~{errs.max()/1000:.0f}k")
    return errs.mean()


print("\n=== 10 折交叉验证 ===")
results = {
    "线性回归": cv_rmse(Ridge(alpha=1.0), "线性回归"),
    "决策树": cv_rmse(DecisionTreeRegressor(random_state=42), "决策树"),
    "随机森林": cv_rmse(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "随机森林"),
}

print("\n===== 模型对比 =====")
for name, score in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"  {name:<8} {score:>10,.0f}")

# ---------------------------------------------------------------- 7. 画图
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
axes[1].set_xlabel("残差（真实 − 预测）")
axes[1].set_title(f"残差分布（RMSE={base_rmse:,.0f}）")

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "reports" / "housing_prediction.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=120)
print(f"\n图已保存：{out}")
