import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso

data = pd.read_csv("house_price_india.csv")

print("DataFrame Columns:", data.columns)

x = data[['living area', 'number of bedrooms', 'number of bathrooms' ]]
y = data['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lr = LinearRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)

# Ridge
ridge = Ridge(alpha=1.0)
ridge. fit(x_train, y_train)
pred_ridge = ridge.predict(x_test)

# Lasso
lasso = Lasso(alpha=0.1)
lasso. fit(x_train, y_train)
pred_lasso = lasso.predict(x_test)

# Evaluation
print("Linear Regression R2:", r2_score(y_test, pred_lr))
print("Ridge R2:", r2_score(y_test, pred_ridge))
print("Lasso R2:", r2_score(y_test, pred_lasso))

area = int(input("Enter Area : "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms:"))

new_data = [[area, bedrooms, bathrooms ]]
new_data = scaler.transform(new_data)

price = ridge.predict(new_data)

print("Predicted House Price:", price[0])