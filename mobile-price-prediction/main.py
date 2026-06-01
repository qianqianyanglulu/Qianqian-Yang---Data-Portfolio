import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.head())
print(test_df.head())

print(train_df.info())

print(train_df["price_range"].value_counts()) 


#可视化：
import seaborn as sns 
import matplotlib.pyplot as plt

sns.violinplot(x="price_range", y="ram", data=train_df)
plt.title("RAM vs Price Range")
# plt.show()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = train_df.drop("price_range", axis=1) 
y = train_df["price_range"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2) 

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train) 

print("Accuracy:", model.score(X_test, y_test)) 


import numpy as np

feature_names = X.columns
for i, class_coef in enumerate(model.coef_):
    print(f"\nClass {i}:")
    for name, coef in zip(feature_names, class_coef):
        print(name, coef)


