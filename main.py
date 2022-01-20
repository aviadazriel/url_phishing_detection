import pandas as pd
from features_extractor import UrlFeaturizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from ModelSummaryTable import ModelSummaryTable
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def build_dataset():
    # https://www.phishtank.com/developer_info.php
    df_phishing = pd.read_csv('phishing_dataset.csv', names=["url"])
    df_phishing = pd.concat([df_phishing, pd.read_csv('phising_tank.csv')[["url"]]])
    df_phishing["label"] = 1
    df_phishing = df_phishing.drop_duplicates()

    # leg urls
    df_leg_link = pd.read_csv("Benign_list_big_final.csv", names=["url"])
    df_leg_link = df_leg_link.sample(n=len(df_phishing), random_state=12).copy()
    df_leg_link = df_leg_link.reset_index(drop=True)
    df_leg_link["label"] = 0

    # concat
    df_phis_leg = pd.concat([df_phishing[["url", "label"]], df_leg_link])
    return df_phis_leg.reset_index()

def preprocessing(df_phis_leg, verbose =0):
    # feature extraction
    print("Begin features extraction")
    url_features = UrlFeaturizer("www.google.com")
    features = url_features.run()
    all_features = []
    columns = list(features.keys())
    df = df_phis_leg.copy()
    for i, row in df.iterrows():
        url = row["url"]
        url_features = UrlFeaturizer(url)
        features = url_features.run()
        all_features.append(list(features.values()))

        if i%2000 ==0 and verbose == 1:
            print(f'(features extraction) {i} iter')

    df[columns] = all_features

    ext_relevant = [key for key, value in Counter(df["ext"]).items() if value > 500]
    df['ext_relevant'] = df['ext'].apply(lambda x: x if x in ext_relevant else 'general')
    df['ext_relevant'] = df['ext_relevant'].astype('category')

    df_ext_onehot_sklearn = df.copy()
    lb = LabelBinarizer()
    lb_results = lb.fit_transform(df_ext_onehot_sklearn['ext_relevant'])
    lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

    result_df = pd.concat([df, lb_results_df], axis=1)
    print("End features extraction")
    data = result_df.sample(frac=1).reset_index(drop=True)
    y = data['label']
    X = data.drop(['label', "index", "url", "ext","ext_relevant"], axis=1)
    print( f'X.shape {X.shape}')
    print(f'y.shape {y.shape}')
    return X, y

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
    tree = DecisionTreeClassifier(max_depth=5)
    # fit the model
    tree.fit(X_train, y_train)
    # predicting the target value from the model for the samples
    y_test_tree = tree.predict(X_test)
    y_train_tree = tree.predict(X_train)
    # computing the accuracy of the model performance
    acc_train_tree = accuracy_score(y_train, y_train_tree)
    acc_test_tree = accuracy_score(y_test, y_test_tree)
    summary_table.add_row('Decision Tree', acc_train_tree, acc_test_tree, 10)
    print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
    print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))

    # Multilayer Perceptrons model
    # instantiate the model
    mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100, 100, 100]))
    # fit the model
    mlp.fit(X_train, y_train)
    # predicting the target value from the model for the samples
    y_test_mlp = mlp.predict(X_test)
    y_train_mlp = mlp.predict(X_train)
    # computing the accuracy of the model performance
    acc_train_mlp = accuracy_score(y_train, y_train_mlp)
    acc_test_mlp = accuracy_score(y_test, y_test_mlp)
    summary_table.add_row('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)
    print("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp))
    print("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp))

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
    acc_train_rf = accuracy_score(y_train, train_conf)
    acc_test_rf = accuracy_score(y_test, test_conf)
    summary_table.add_row('Random Forest', acc_train_rf, acc_test_rf, clf.best_params_['max_depth'], clf.best_params_['n_estimators'])
    print("Random Forest: Accuracy on training Data: {:.3f}".format(acc_train_rf))
    print("Random Forest: Accuracy on test Data: {:.3f}".format(acc_test_rf))

    summary_table.show()






