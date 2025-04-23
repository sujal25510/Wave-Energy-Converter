import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import os

data = pd.read_csv('/kaggle/input/waveenergy/Sydney_Data.csv')
print(data.shape)

# Selecting top 1k rows
data = data[:1000]
print(data.shape)
column_names = [f'X{i}' for i in range(1, 17)] + [f'Y{i}' for i in range(1, 17)] + [f'P{i}' for i in range(1, 17)] + ['Power_Output']

# Assign the new column names to the DataFrame
data.columns = column_names

# os.listdir('/kaggle/input')
# ['Sydney_Data.csv', 'Perth_Data.csv', 'Adelaide_Data.csv', 'Tasmania_Data.csv']


sample_data = data.sample(1000)
cat_columns = sample_data.select_dtypes(object).columns.to_list()
print(cat_columns)
print()
num_columns = sample_data.select_dtypes(np.number).columns.to_list()
print(num_columns)


s = MinMaxScaler()
data = pd.DataFrame(s.fit_transform(data), index= data.index, columns= data.columns)


X = data.iloc[:, :-1]  # Assuming the target variable is the last column
y = data.iloc[:, -1]   # Assuming the target variable is the last column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Machine": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor()
}



# Define the hyperparameters grid for each model
param_grids = {
    "Linear Regression": {},
    "Decision Tree": {"max_depth": [None, 5, 10, 15]},
    "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 15]},
    "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7]}
}

# Perform GridSearchCV for each model
best_models = {}

for model_name, model in models.items():
    print("Tuning hyperparameters for", model_name)
    
    # Create GridSearchCV instance
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], scoring="neg_mean_squared_error", cv=5)
    
    # Fit the GridSearchCV
    grid_search.fit(X_train, y_train)
     # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Save the best model and its parameters
    best_models[model_name] = {"model": best_model, "params": best_params}
    
    # Print the best parameters and best score
    print("Best parameters:", best_params)
    print("Best score:", grid_search.best_score_)
    print()

rmse_scores = {}

for model_name, model_data in best_models.items():
    model = model_data["model"]
    params = model_data["params"]
    
    print("Evaluating", model_name)
    print("Best parameters:", params)
    
    # Fit the model with best parameters
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores[model_name] = rmse
    
    print("Root Mean Squared Error:", rmse)
    print()

plt.figure(figsize=(10, 6))
plt.bar(rmse_scores.keys(), rmse_scores.values())
plt.xlabel("Regression Models")
plt.ylabel("RMSE")
plt.title("Root Mean Squared Error for Different Regression Models")
plt.xticks(rotation=45)
plt.show()

