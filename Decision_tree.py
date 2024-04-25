import pandas as pd
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names,
                                     filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)

data = pd.DataFrame({'A': [17,64,18,20,38,49,55,25,29,31,33],
             'B': [18,65,19,21,39,50,56,26,30,32,34],
             'Success': [1,0,1,0,1,0,0,1,1,0,1]})
data = data.sort_values('A')

#define Decision Tree
dt = DecisionTreeClassifier()
#define input vectors
#X is the features in this dataset
X = data.iloc[:, :-1]
#Y is the vector with our Target Variables
Y = data.iloc[:, -1]
#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=123)
#start fitting process
dt.fit(X_train, y_train)

tree_graph_to_png(dt, feature_names=['A', 'B'], png_file_to_save='dt.png')
X_test
y_test
dt.predict(X_test)

