from utils import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ModelSummaryTable import ModelSummaryTable
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    # prepare data
    dataset = build_dataset()
    X, y = preprocessing(dataset)
    # Splitting the dataset into train and test sets: 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # Models
    summary_table = ModelSummaryTable()

    # Decision Tree model
    # instantiate the model
    max_depth = 5
    tree = DecisionTreeClassifier(max_depth=max_depth)
    # fit the model
    tree.fit(X_train, y_train)
    # predicting the target value from the model for the samples
    y_test_tree = tree.predict(X_test)
    y_train_tree = tree.predict(X_train)
    # computing the accuracy of the model performance
    model_name = 'Decision Tree'
    acc_train_tree, acc_test_tree = calculate_accuracy(y_train, y_train_tree, y_test, y_test_tree, model_name)
    summary_table.add_row(model_name, acc_train_tree, acc_test_tree)

    # Multilayer Perceptrons model
    # instantiate the model
    mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100, 100, 100]))
    # fit the model
    mlp.fit(X_train, y_train)
    # predicting the target value from the model for the samples
    y_test_mlp = mlp.predict(X_test)
    y_train_mlp = mlp.predict(X_train)
    # computing the accuracy of the model performance
    model_name = 'Multilayer Perceptrons'
    acc_train_mlp, acc_test_mlp = calculate_accuracy(y_train, y_train_mlp, y_test, y_test_mlp, model_name)
    summary_table.add_row(model_name, acc_train_mlp, acc_test_mlp)

    # Random Forest
    # instantiate the model
    param_grid = {
        'max_depth': [4, 8, 16],
        'n_estimators': [1, 2, 5, 10]
    }
    rf = RandomForestClassifier(n_jobs=-1)
    clf = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='roc_auc')
    # fit thr model
    clf.fit(X_train, y_train)
    rf_clf = RandomForestClassifier(max_depth=clf.best_params_['max_depth'],
                                    n_estimators=clf.best_params_['n_estimators'])
    rf_clf.fit(X_train, y_train)
    # Making predictions
    test_conf = rf_clf.predict(X_test)
    train_conf = rf_clf.predict(X_train)

    # computing the accuracy of the model performance
    model_name = 'Random Forest'
    acc_train_rf, acc_test_rf = calculate_accuracy(y_train, train_conf, y_test, test_conf, model_name)
    summary_table.add_row(model_name, acc_train_rf, acc_test_rf, clf.best_params_['max_depth'], clf.best_params_['n_estimators'])

    # Summary
    summary_table.show()






