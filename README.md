# ml-practice

Python 机器学习练习记录。以 notebook 为主，重点在理解算法和推导过程，不追求工程化。

## 环境

- Python 3.13（由 uv 固定，见 `.python-version`）
- 依赖由 uv 管理，**环境不入库**，克隆后一条命令重建：

```bash
uv sync                      # 重建虚拟环境
uv run jupyter lab           # 在浏览器里打开
```

装了 PyCharm 的话，用 `File → Open` 打开本目录，它会自动识别 `.venv`；
打开 `.ipynb` 后右上角 kernel 选 **Python (ml-practice)**。

手动安装过的包请用 `uv add <包名>` 加进 `pyproject.toml`，不要直接 pip install，
否则换机器时 `uv sync` 装不全。

## 目录

```
notebooks/   练习 notebook，按主题分文件
src/         可复用的函数（特征工程、评估、画图）
data/        数据集（不进 git，见 data/README.md）
models/      模型权重（不进 git）
reports/     图表与报告产出
```

## 练习记录

<!-- 每完成一个练习加一行，倒序排列 -->

| 日期 | 主题 | 数据集 | notebook | 备注 |
|---|---|---|---|---|
| | 分类流程入门 | 鸢尾花 (150条) | [01_iris_classification](notebooks/01_iris_classification.ipynb) | 五步流程 + 混淆矩阵 |
| | 回归与过拟合 | California 房价 | [02_linear_regression](notebooks/02_linear_regression.ipynb) | 多项式阶数 / 正则化对比 |
| | 神经网络 | 手写数字 (8×8) | [03_digits_torch](notebooks/03_digits_torch.ipynb) | sklearn MLP 与 PyTorch 训练循环 |
| | 加州房价（迁移自 Windows 旧项目） | housing.csv | [src/housing_price_forecasting.py](src/housing_price_forecasting.py) | 仿原书第 2 章，修 5 处 API 变更 |

## 参考

- 《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》，Aurélien Géron 著，
  O'Reilly。`src/housing_price_forecasting.py` 参考该书第 2 章加州房价项目改写，
  原书代码见 <https://github.com/ageron/handson-ml2>（Apache License 2.0）
- California housing 数据集：源自 StatLib，1990 年美国人口普查，Pace & Barry (1997) 整理

## 约定

- ⚠️ 提交 notebook 前先清输出：`jupyter nbconvert --clear-output --inplace notebooks/xxx.ipynb`
