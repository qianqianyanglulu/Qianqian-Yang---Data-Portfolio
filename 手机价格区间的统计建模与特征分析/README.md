# 手机价格预测项目（Mobile Price Classification）

## 项目简介

这个项目的目标是通过手机的各项参数（RAM、电池、屏幕等）来预测价格区间。

一开始我只是想做一个基础的机器学习练习，但在过程中逐渐发现，这个数据集其实很适合用来练习从直觉 --> 验证 --> 优化模型的完整流程。于是我决定把这个过程系统的记录下来。

---

## 🧠 我的思考过程

### 1️⃣ 从直觉出发：RAM 会不会决定价格？

在没有建模之前，我的第一反应是：手机价格最直观的决定因素应该是RAM

所以我先画了一个最简单的图：

```python
plt.scatter(train_df["ram"], train_df["price_range"])
```

但很快发现 price_range 是离散型数据，所有点几乎连成一条直线，没有达到可视化“清晰”的目标。于是我决定进行优化。



### 2️⃣ 改用 Seaborn 的 boxplot

之所以考虑用 boxplot 是因为它有几个特点刚好能解决 scatter 的问题：

* 可以按类别（price_range）分组
* 能直接看到中位数（典型水平）
* 能看到分布范围（高低差异）

所以我换成：

```python
sns.boxplot(x="price_range", y="ram", data=train_df)
```

通过这一步我们就能在图上清晰的看到 price 与 RAM 之间的联系。



### 3️⃣ 第一次建模：选择了 Logistic Regression

在模型选择上，我一开始选择用 Logistic Regression 是因为这是一个分类问题（price_range 是 0–3）并且Logistic Regression 是最基础、最可解释的分类模型。因此决定先用一个“baseline”，通过 Accuracy 再考虑如何进行下一步优化。
在 Logistic Rregression 模型下 Accuracy ≈ 0.69
这个结果其实有点低，但也在预期之内。


### 4️⃣ 为什么后来加了标准化（StandardScaler）？

我开始反思：不同特征的尺度差异很大
比如：
* RAM 是几千
* clock_speed 是 0–3

这会影响模型学习权重。于是我做了一个很典型的改进：

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

结果非常明显：Accuracy 从 0.69 --> 提升到 0.96

这一步让我真正理解：数据预处理，有时候比换模型更重要



### 5️⃣ 一些观察

最后我查看了模型的系数（feature importance）：发现 RAM 是影响价格最强的因素（远高于其他特征）

同时对手机价格影响强的因素还有：

* 屏幕分辨率（px_width / px_height）
* 电池容量（battery_power）

感觉这和现实是一致的。



## 🍓 最终结果

* 模型：Logistic Regression + StandardScaler
* Accuracy：**约 96%**



## 🍓 收获

这个项目让我意识到几件事：

* 可视化不是“画图”，而是为了回答问题
* 简单模型 + 正确处理，效果可以很好
* 从直觉出发，再用数据验证，是一个很自然的路径



## 🛠️ 技术栈

Python / Pandas / Scikit-learn / Matplotlib / Seaborn

## 🛠️ Dataset 来源

kaggle -- Mobile Price Classification
