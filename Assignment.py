# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # House Sales Price Prediction
# #### An Advnaced Regression Problem
#
# Eeshan Gupta  
# eeshangpt@gmail.com

# %% [markdown]
# ## Table of Content
# 1. [Introduction to the problem]()
#     1. [Business Understanding]()
#     2. [Business Goal]()
# 1. [Reading and Cleaning the Data]()
#     1. [Data Dictionary]()
#     2. [Missing Value Analysis]()
#     3. [Cleaning the Data]()
# 1. <>
#     1. rkld;f
# 1. <>
#     1. rkld;f
# 1. <>
#     1. rkld;f
# 1. <>
#     1. rkld;f
# 1. <>
#     1. rkld;f
# 1. <>
#     1. rkld;f

# %% [markdown]
# ## Introduction to the problem
# ### Business Understanding
#
# * A US-based housing company named Surprise Housing has decided to enter the Australian market
# * The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price
# * For the same purpose, the company has collected a data set from the sale of houses in Australia
#
# ### Business Goal
#
# * Which variables are significant in predicting the price of a house
# * How well those variables describe the price of a house

# %% [markdown]
# Imports and Standard Settings

# %%
import warnings
from os import getcwd
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, Normalizer

# %%
np.random.seed(42)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_colwidth', None)
sns.set_style('darkgrid')

# %%
PRJ_DIR = getcwd()
DATA_DIR = join(PRJ_DIR, "data")

# %% [markdown]
# ## Reading and Cleaning the Data

# %%
df = pd.read_csv(join(DATA_DIR, "train.csv"))
df.head()

# %%
df.info()


# %% [markdown]
# ### Data Dictionary

# %%
def get_data_dictionary():    
    with open(join(DATA_DIR, "data_description.txt"), 'r') as f:
        a = f.readlines()
    
    data_dict = {}
    for itr, line in enumerate(a):
        _ = line.split(' ')
        if _[0] not in ['', '\n']:
            column_name = _[0].split(':')[0]
            data_dict[column_name] = {"meaning" : [], "categories":{}}
            meaning = " ".join(_[1:]).strip()
            data_dict[column_name]["meaning"] = meaning
        elif _[0] == '\n':
            continue
        elif _[0] == '':
            line = " ".join([i for i in _ if i != ''])
            temp = line.strip().split("\t")
            if len(temp) > 1:
                cat_name, cat_meaning = temp[0], temp[1]
            data_dict[column_name]['categories'][cat_name] = cat_meaning
    data_dict = {k: v for k, v in data_dict.items() if v['meaning'] != ''}
    
    numerical_columns, categorical_columns = [], []
    for (k, v) in data_dict.items():
        if v['meaning'] != '':
            if len(v['categories']) == 0:
                numerical_columns.append(k)
            if len(v['categories']) > 0:
                categorical_columns.append(k)
    pprint(data_dict)
    return categorical_columns, numerical_columns, data_dict


# %%
categorical_columns, numerical_columns, _ = get_data_dictionary()

# %% [markdown]
# ### Missing Value Analysis

# %%
(df.isna().sum(axis=0) > 0).sum()

# %% [markdown]
# Removing columns with any missing values

# %%
empty_columns = df.columns[(df.isna().sum(axis=0))>0]
df = df[[i for i in df.columns if i not in empty_columns]]
df.shape

# %%
all_unique_columns = df.columns[df.nunique() == df.shape[0]]

# %%
df = df[[i for i in df.columns if i not in all_unique_columns]]

# %%
df.shape

# %%
df.head()

# %%
df.info()

# %% [markdown]
# ### Cleaning the data

# %%
X = df.copy()
Y = X.pop('SalePrice').astype(float)

# %% [markdown]
# #### Feature Variables

# %%
date_columns = ["YearBuilt", "YearRemodAdd", "YrSold", "MoSold", "MSSubClass",]

# %%
X.info()

# %%
categorical_columns = [i for i in X.columns if i in categorical_columns]
numerical_columns = [i for i in X.columns if i in numerical_columns]
other_columns = [i for i in X.columns if i not in numerical_columns + categorical_columns]

# %% [markdown]
# Columns not present in the data dictionary

# %%
X[other_columns].nunique()

# %%
for _ in other_columns:
    print(_, X[_].unique(), sep='\n', end='\n==\n')

# %% [markdown]
# Since the total number of unique values are quite low, therefore deeming them to be categorical variables

# %%
categorical_columns += other_columns
del other_columns

# %% [markdown]
# Adding date type columns to categorical variables

# %%
categorical_columns += date_columns

# %% [markdown]
# Defining the categorical and numerical columns

# %%
categorical_columns = list(set(categorical_columns))
numerical_columns = list(set([i for i in numerical_columns if i not in date_columns]))

# %% [markdown]
# Cleaning the date type and creating the age

# %%
date_columns = date_columns[:-1]

month_dict = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',
}

X['MoSold'] = X.MoSold.astype(int).apply(lambda x: month_dict[x]).astype(str)
X['BuiltAge'] = (2023 - X.YearBuilt.astype(int))
X['LastRemod'] = (2023 - X.YearRemodAdd.astype(int))
X['LastSold'] = (2023 - X.YrSold.astype(int))

for i in ["YearBuilt", "YearRemodAdd", "YrSold",]:
    categorical_columns.remove(i)

numerical_columns += ['BuiltAge', 'LastRemod', 'LastSold']
del date_columns

# %%
X[numerical_columns].describe()

# %%
X[numerical_columns].info()

# %%
numerical_description = X[numerical_columns].describe()
other_categorical_columns = [i for i in numerical_columns if i not in numerical_description.columns]
numerical_columns = numerical_description.columns.to_list() + ['1stFlrSF']
categorical_columns += other_categorical_columns
categorical_columns.remove('1stFlrSF')
del other_categorical_columns
numerical_description

# %%
X[categorical_columns].info()

# %%
X[categorical_columns] = X[categorical_columns].astype(str)
X[categorical_columns].info()

# %%
X[numerical_columns].info()

# %% [markdown]
# #### Target Variables (Sales Price)

# %%
sns.distplot(Y)

# %% [markdown]
# The target varaiable is distributed normally but skewed. Taking log of the varaible removes the skewness in the distribution

# %%
sns.distplot(np.log(Y))

# %% [markdown]
# ## Exploratory Data Analysis
# Defining target variable

# %%
Y_log = np.log(Y)

# %% [markdown]
# ### Correlation with numerical variables

# %%
plt.figure(figsize=(20,20))
a = pd.concat((X[numerical_columns], Y_log), axis=1)
sns.heatmap(a.corr(), annot=True)
del a
plt.show()

# %% [markdown]
# Finding the correlation of numerical varibles with target variable

# %%
corr_salesprice = X[numerical_columns].corrwith(Y_log).sort_values(ascending=False).reset_index()
corr_salesprice.columns = ['col_name', 'corr_value']
corr_salesprice

# %%
highly_corr_columns = corr_salesprice[(corr_salesprice.corr_value >= 0.5) | (corr_salesprice.corr_value <= -0.5)]
highly_corr_columns

# %%
plt.figure()
a = pd.concat((X[highly_corr_columns.col_name], Y_log), axis=1)
sns.pairplot(data=a)
del a
plt.show()

# %% [markdown]
# ### Bivariate Analysis of categorical variable with target variable

# %%
# fig, axs = plt.subplots(36, 1, figsize=(20, 6))
for itr, i in enumerate(categorical_columns):
    a = X[i].nunique()
    if a < 4:
        a = 4
    plt.figure(figsize=(a, a))
    sns.boxplot(x=X[i], y=Y_log)
    plt.title(f"{str(i).upper()}")
    plt.xticks(rotation=90)
    plt.show()

# %% [markdown]
# ### Data Preparation

# %%
X_cat = pd.get_dummies(X[categorical_columns], drop_first=True, prefix=[f"{i}_" for i in categorical_columns]).astype(int)
X_num = X[numerical_columns]
X_final = pd.concat((X_cat, X_num), axis=1)
del X_cat, X_num
X_final.head()

# %% [markdown]
# Final shape of the training data

# %%
X_final.shape 

# %%
final_columns = X_final.columns.to_list()

# %%
Y_final = Y_log.copy()
del Y_log
Y_final.head()

# %% [markdown]
# ## Data Preparation for model training

# %% [markdown]
# ### Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(X_final, Y_final, train_size=0.7, random_state=42)

# %% [markdown]
# ### Scaling the data

# %%
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=final_columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=final_columns)

# %% [markdown]
# Using standard scaling since it is providing better results

# %% [raw]
# mean_y, std_y = y_train.mean(), y_train.std()
#
# y_train = (y_train - mean_y) / std_y
# y_test = (y_test - mean_y) / std_y

# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
sns.distplot(y_train)

# %%
X_train.head()

# %% [markdown]
# ## Linear Regression Model Training

# %%
slr_estimator = LinearRegression()

# %%
rfe_model = RFE(slr_estimator, n_features_to_select=0.5)
rfe_model = rfe_model.fit(X_train, y_train)

# %%
rfe_analysis = pd.DataFrame(list(zip(final_columns, rfe_model.support_, rfe_model.ranking_)), columns=['ColumnName', 'Support', 'Rank'])
rfe_analysis.sort_values(by='Rank')

# %%
selected_colums = rfe_analysis[rfe_analysis.Support]['ColumnName']
selected_colums

# %%
X_train[selected_colums].head()

# %%
X_train_subset = X_train[selected_colums]
X_test_subset = X_test[selected_colums]

# %%
lr_model = LinearRegression()
lr_model.fit(X_train_subset, y_train)

# %%
lr_model.coef_

# %%
y_tr_pred = lr_model.predict(X_train_subset)
print(f"R2 score for the training data is {r2_score(y_train, y_tr_pred)}")
print(f"MSE for the training data is {mean_squared_error(y_train, y_tr_pred)}")

# %%
y_ts_pred = lr_model.predict(X_test_subset)
print(f"R2 score for the test data is {r2_score(y_test, y_ts_pred)}")
print(f"MSE for the test data is {mean_squared_error(y_test, y_ts_pred)}")

# %%
lr_model.coef_

# %% [markdown]
# The $R^2$ value for the test is negative, I am choosing to keep all the columns for Ridge and Lasso Regression

# %% [markdown]
# ## Ridge and Lasso Regression Models

# %% [markdown]
# ### Parameters
#
# Setting up parameters for the GridSearch

# %% [markdown]
# Number of folds

# %%
folds = 5

# %% [markdown]
# Possible value of alphas

# %%
alphas = sorted(set(np.linspace(0, 1, 9).tolist() + np.linspace(0, 0.125, 9).tolist() + np.linspace(1, 10, 9).tolist() + np.linspace(10, 100, 10).tolist() + np.linspace(100, 1000, 10).tolist()))
params = {'alpha': alphas}
pprint(params)

# %% [markdown]
# ### Ridge Regression

# %%
ridge_lr = Ridge()
ridge_lr_cv = GridSearchCV(estimator=ridge_lr, param_grid=params, scoring='r2', cv=folds, return_train_score=True, verbose=5)
ridge_lr_cv.fit(X_train, y_train)

# %%
ridge_lr_cv.best_params_

# %%
ridge_lr_cv.best_score_

# %%
ridge_lr_best_model = Ridge(alpha=ridge_lr_cv.best_params_['alpha'])
ridge_lr_best_model.fit(X_train, y_train)

# %%
X_train.shape

# %%
coeff_df = pd.DataFrame(list(zip(X_train.columns, ridge_lr_best_model.coef_)), columns=['ColumnName', 'Coefficient'])
coeff_df.sample(15)

# %%
y_tr_pred = ridge_lr_best_model.predict(X_train)

# %%
print(f"R2 score for the training data is {r2_score(y_train, y_tr_pred)}")
print(f"MSE for the training data is {mean_squared_error(y_train, y_tr_pred)}")

# %%
y_ts_pred = ridge_lr_best_model.predict(X_test)

# %%
print(f"R2 score for the test data is {r2_score(y_test, y_ts_pred)}")
print(f"MSE for the test data is {mean_squared_error(y_test, y_ts_pred)}")

# %% [markdown]
# ### Lasso Regression

# %%
lasso_lr = Lasso()
lasso_lr_cv = GridSearchCV(estimator=lasso_lr, param_grid=params, scoring='r2', cv=folds, return_train_score=True, verbose=5)
lasso_lr_cv.fit(X_train, y_train)

# %%
lasso_lr_cv.best_params_

# %%
lasso_lr_cv.best_score_

# %%
lasso_lr_best_model = Lasso(alpha=lasso_lr_cv.best_params_['alpha'])
lasso_lr_best_model.fit(X_train, y_train)

# %%
coeff_df = pd.DataFrame(list(zip(X_train.columns, lasso_lr_best_model.coef_)), columns=['ColumnName', 'Coefficient'])
coeff_df.sample(15)

# %%
y_tr_pred = lasso_lr_best_model.predict(X_train)

# %%
print(f"R2 score for the training data is {r2_score(y_train, y_tr_pred)}")
print(f"MSE for the training data is {mean_squared_error(y_train, y_tr_pred)}")

# %%
y_ts_pred = lasso_lr_best_model.predict(X_test)

# %%
print(f"R2 score for the test data is {r2_score(y_test, y_ts_pred)}")
print(f"MSE for the test data is {mean_squared_error(y_test, y_ts_pred)}")

# %%
