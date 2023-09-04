import pandas as pd
# implement RandomForestClassifier from sklearn - https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# need to upload income.csv first
df = pd.read_csv('income.csv')
df


# do the one-hot encoding, adding prefix and dropping on other features (except fot the gender column)
# do this beacuse we want the dataset to be numerical or binary features
df = pd.concat([df.drop('occupation', axis=1), pd.get_dummies(df.occupation).add_prefix('occupation_')], axis=1)
df = pd.concat([df.drop('workclass', axis=1), pd.get_dummies(df.workclass).add_prefix('workclass_')], axis=1)
df = df.drop('education', axis=1) # we have educational-num and it's enough, so we drop the education column
df = pd.concat([df.drop('marital-status', axis=1), pd.get_dummies(df['marital-status']).add_prefix('marital-status_')], axis=1)
df = pd.concat([df.drop('relationship', axis=1), pd.get_dummies(df.relationship).add_prefix('relationship_')], axis=1)
df = pd.concat([df.drop('race', axis=1), pd.get_dummies(df.race).add_prefix('race_')], axis=1)
df = pd.concat([df.drop('native-country', axis=1), pd.get_dummies(df['native-country']).add_prefix('native-country_')], axis=1)

# encode the gender and the income to be binary (0 or 1)
# pandas.DataFrame.apply() - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
# apply(function)
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)


# Training
df = df.drop('fnlwgt', axis=1) # This fnlwgt feature is not a decent feature to our model

# Split the data into train_df and test_df
train_df, test_df = train_test_split(df, test_size=0.2)

# Separate the data into X and y (features and label), which is other features and income
# drop the income column
train_X = train_df.drop('income', axis=1)
train_y = train_df['income']

test_X = test_df.drop('income', axis=1)
test_y = test_df['income']


# RandomForestClassifier() - https://ithelp.ithome.com.tw/articles/10272586
forest = RandomForestClassifier()
forest.fit(train_X, train_y) # train

# evaluate with test data, return accuracy
forest.score(test_X, test_y)
