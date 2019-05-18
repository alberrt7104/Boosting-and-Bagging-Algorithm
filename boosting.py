import pandas as pd
df_wine = pd.read_csv('outputrain', header=None)


from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
le = LabelEncoder()





from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)

tree_train = accuracy_score(y_train, y_train_pred)

print('Decision tree train %.3f' % (tree_train)) 
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=50, learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)

ada_train = accuracy_score(y_train, y_train_pred)

print('AdaBoost train %.3f' % (ada_train)) 
