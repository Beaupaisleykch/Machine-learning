# data/

数据集**不进 git**（已在 .gitignore 里排除）。这里只放说明：

| 文件名 | 来源 | 大小 | 说明 |
|---|---|---|---|
| （示例）iris.csv | sklearn 内置 / UCI | 4KB | 鸢尾花分类练习 |
| housing/housing.csv | [ageron/handson-ml2](https://github.com/ageron/handson-ml2) `datasets/housing/housing.tgz` | 1.4MB | California housing，源自 StatLib / 1990 年美国人口普查，Pace & Barry (1997) |

约定：
- 小数据集（<5MB）可以从这里放，但仍不建议提交，改为在 notebook 里用 `sklearn.datasets.load_*`
- 大数据集写清下载地址和校验方式，别人 clone 后能自行获取
