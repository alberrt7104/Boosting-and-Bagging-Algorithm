import pandas as pd
df_wine = pd.read_csv('outputrain', header=None)


from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
le = LabelEncoder()



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)

bag = BaggingClassifier(base_estimator=tree, n_estimators=50, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)

tree_train = accuracy_score(y_train, y_train_pred)

print('Decision tree train %.3f' % (tree_train))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)

bag_train = accuracy_score(y_train, y_train_pred)

print('Bagging trainaccuracies %.3f' % (bag_train))
