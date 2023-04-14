import pandas as pd
import numpy as np

df = pd.read_csv("jsresults2.csv", sep=',', delimiter=None, 
                header='infer', names=None, 
                index_col=None, usecols=None, squeeze=False,engine=None)

df = df.dropna(subset=['Satisfaction'])
X = df.drop("Satisfaction", axis = 1)
y = df['Satisfaction']

# Create two new columns "Man" and "Woman" with default values of 0
df.insert(3, "Man", 0)
df.insert(4, "Woman", 0)
# Create two new columns "Bachelor" and "Master" with default values of 0
df.insert(6, "Bachelor", 0)
df.insert(7, "Master", 0)

# Loop through each row in the dataframe
for index, row in df.iterrows():
    # Check the value of the "gender" column in the current row
    if row["Gender"] == "Man":
        # If the surveyee is a man, set the "Male" column to 1
        df.at[index, "Man"] = 1
    elif row["Gender"] == "Woman":
        # If the surveyee is a woman, set the "Female" column to 1
        df.at[index, "Woman"] = 1

# Loop through each row in the dataframe
for index, row in df.iterrows():
    # Check the value of the "education" column in the current row
    if "bachelor" in row["Education level"].lower():
        # If the surveyee has a bachelor's degree, set the "Bachelor" column to 1
        df.at[index, "Bachelor"] = 1
    elif "master" in row["Education level"].lower():
        # If the surveyee has a master's degree, set the "Master" column to 1
        df.at[index, "Master"] = 1

# Reversed questions (works while questions are numbered like Q1, Q2, Q3)
negatively_worded_items = [2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 21, 23, 24, 26, 29, 31, 32, 34, 36]
negatively_worded_items = ['Q' + str(x) for x in negatively_worded_items]

reversal_map = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}

for item in negatively_worded_items:
    column_index = df.columns.get_loc(item)
    df.iloc[:, column_index] = df.iloc[:, column_index].map(reversal_map)

# Creating columns of dimensions with corresponding questions
df["Pay"] = df.iloc[:, 9:10].sum(axis=1) + df.iloc[:, 18:19].sum(axis=1) + df.iloc[:, 27:28].sum(axis=1) + df.iloc[:, 36:37].sum(axis=1)
df["Promotion"] = df.iloc[:, 10:11].sum(axis=1) + df.iloc[:, 19:20].sum(axis=1) + df.iloc[:, 28:29].sum(axis=1) + df.iloc[:, 41:42].sum(axis=1)
df["Supervision"] = df.iloc[:, 11:12].sum(axis=1) + df.iloc[:, 20:21].sum(axis=1) + df.iloc[:, 29:30].sum(axis=1) + df.iloc[:, 38:39].sum(axis=1)
df["Fringe Benefits"] = df.iloc[:, 12:13].sum(axis=1) + df.iloc[:, 21:22].sum(axis=1) + df.iloc[:, 30:31].sum(axis=1) + df.iloc[:, 37:38].sum(axis=1)
df["Contingent Rewards"] = df.iloc[:, 13:14].sum(axis=1) + df.iloc[:, 22:23].sum(axis=1) + df.iloc[:, 31:32].sum(axis=1) + df.iloc[:, 40:41].sum(axis=1)
df["Operating Procedures"] = df.iloc[:, 14:15].sum(axis=1) + df.iloc[:, 23:24].sum(axis=1) + df.iloc[:, 32:33].sum(axis=1) + df.iloc[:, 39:40].sum(axis=1)
df["Co-workers"] = df.iloc[:, 15:16].sum(axis=1) + df.iloc[:, 24:25].sum(axis=1) + df.iloc[:, 33:34].sum(axis=1) + df.iloc[:, 42:43].sum(axis=1)
df["Nature of Work"] = df.iloc[:, 16:17].sum(axis=1) + df.iloc[:, 25:26].sum(axis=1) + df.iloc[:, 35:36].sum(axis=1) + df.iloc[:, 43:44].sum(axis=1)
df["Communication"] = df.iloc[:, 17:18].sum(axis=1) + df.iloc[:, 26:27].sum(axis=1) + df.iloc[:, 34:35].sum(axis=1) + df.iloc[:, 43:44].sum(axis=1)


# Pay for 1st after reverse should be = 2 + 7-2 + 7-3 + 4 = 15
# Promotion for 1st after reverse should be = 7-3 + 4 + 5 + 5 = 18
# Supervision for 1st after reverse should be = 5 + 7-2 + 7-4 + 5 = 18

from sklearn.model_selection import train_test_split

# Removing redundant columns
droplist=[]
for i in range (1,37):
  droplist.append('Q'+str(i))
droplist.append('ID')
droplist.append('Gender')
droplist.append('Education level')
droplist.append('Satisfaction')

# trying to get better score
droplist.append('Man')
droplist.append('Woman')
droplist.append('Bachelor')
droplist.append('Master')
droplist.append('Age')
# droplist.append('Co-workers')
X = df
X = X.drop(droplist, axis=1)

# Adding random feature
# np.random.seed(seed = 44)
# X['random'] = np.random.random(size = len(X))

# 1.1 Default Sci-kit feature importance method
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Fit the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# Evaluate the accuracy of the model on the training and test sets
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Print the accuracy scores
print("Training R^2 score: {:.3f}".format(train_r2))
print("Training MSE score: {:.3f}".format(train_mse))
print("Test R^2 score: {:.3f}".format(test_r2))
print("Test MSE score: {:.3f}".format(test_mse))

# Calculate the feature importance scores
coef = regressor.coef_
abs_coef = np.abs(coef)
feature_importance = 1 * (abs_coef / np.sum(abs_coef))
print()
# Print the feature importances
for i in sorted(range(len(feature_importance)), key=lambda k: abs(feature_importance[k]), reverse=True):
    print(f"{X_train.columns[i]}: {round(feature_importance[i],3)}")
sorted_idx = feature_importance.argsort()
# Plot the feature importance scores
plt.figure(figsize=(8, 5))
# plt.title("Linear Regression Feature Importances")
plt.barh(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), X.columns[sorted_idx])
plt.xlabel("Linear Regression Feature Importances")
plt.ylim([-1, X.shape[1]])
plt.show()

#1.2 Permutation feature importance method
from sklearn.inspection import permutation_importance

result_test = permutation_importance(
    regressor, X, y, n_repeats=20, random_state=42, n_jobs=2
)
sorted_importances_idx_test = result_test.importances_mean.argsort()
importances_test = pd.DataFrame(
    result_test.importances[sorted_importances_idx_test].T,
    columns=X.columns[sorted_importances_idx_test],
)

#calculate permutation importance for training data 
result_train = permutation_importance(
    regressor, X_train, y_train, n_repeats=20, random_state=42, n_jobs=2,
)

sorted_importances_idx_train = result_train.importances_mean.argsort()
importances_train = pd.DataFrame(
    result_train.importances[sorted_importances_idx_train].T,
    columns=X.columns[sorted_importances_idx_train],)

#In permutation feature importance, the feature importance scores do not add up to 100%, but they show the relative importance of each feature in the model. 
#The sum of the feature importances can be greater than 1 because each feature is measured independently, so the importance of one feature does not affect the importance of another feature.


f, axs = plt.subplots(1,2,figsize=(15,5))

importances_test.plot.box(vert=False, whis=10, ax = axs[0])
axs[0].set_title("Permutation Importances")
axs[0].axvline(x=0, color="k", linestyle="--")
axs[0].set_xlabel("Decrease in accuracy score")
axs[0].figure.tight_layout()

importances_train.plot.box(vert=False, whis=10, ax = axs[1])
axs[1].set_title("Permutation Importances (train set)")
axs[1].axvline(x=0, color="k", linestyle="--")
axs[1].set_xlabel("Decrease in accuracy score")
axs[1].figure.tight_layout()

# Print the feature importances
for i in result_test.importances_mean.argsort()[::-1]:
    print(f"{X.columns[i]:<20} "
          f"{result_test.importances_mean[i]:.3f}"
          f" +/- {result_test.importances_std[i]:.3f}")

#1.3 Drop column feature importance method

# Compute the performance of the model on the testing set
y_pred_test = regressor.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)

importances = []
for col in X.columns:
    X_drop = X.drop(col, axis=1)
    X_train_drop, X_val_drop, y_train_drop, y_val_drop = train_test_split(X_drop, y, test_size=0.3, random_state=42)
    reg_drop = LinearRegression()
    reg_drop.fit(X_train_drop, y_train_drop)
    y_val_pred_drop = reg_drop.predict(X_val_drop)
    mse_drop = mean_squared_error(y_val_drop, y_val_pred_drop)
    importances.append(mse_drop - mse_test)

# Create a dataframe of feature importances
importances_df = pd.DataFrame({'feature': X.columns, 'importance': importances})

# Sort the dataframe by feature importance
importances_df = importances_df.sort_values('importance', ascending=True)

# Print the feature importances
print(importances_df)

plt.barh(importances_df['feature'], importances_df['importance'])

# Plot all three methods in one graph
ardef = feature_importance.tolist()
dftotal = pd.DataFrame(X_test.columns,columns=['Feature'])
dftotal['Default'] = ardef
dftotal["Permutation"] = result_test.importances_mean
dftotal['Drop-column'] = importances

dftotal = dftotal.sort_values(by=['Permutation'], ascending=True)

dftotal.plot(kind = 'barh', y=['Default','Permutation','Drop-column'], x = 'Feature', )

plt.show()
