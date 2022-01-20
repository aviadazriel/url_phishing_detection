import pandas as pd
from features_extractor import UrlFeaturizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from ModelSummaryTable import ModelSummaryTable

def preprocessing(phis_file_path , lef_file_path):
    # https://www.phishtank.com/developer_info.php
    df_phishing = pd.read_csv(phis_file_path)
    df_phishing["label"] = 1

    # leg urls
    df_leg_link = pd.read_csv(lef_file_path, names=["url"])
    df_leg_link = df_leg_link.sample(n = len(df_phishing), random_state = 12).copy()
    df_leg_link = df_leg_link.reset_index(drop=True)
    df_leg_link["label"] = 0

    # concat
    df_phis_leg = pd.concat([df_phishing[["url", "label"]], df_leg_link])
    df_phis_leg = df_phis_leg.reset_index()

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

        if i%2000 ==0 :
            print(f'(features extraction) {i} iter')

    df[columns] = all_features

    ext_relevant = [key for key, value in Counter(df["ext"]).items() if value > 100]
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
    X, y = preprocessing('./phising_tank.csv','./Benign_list_big_final.csv')
    # Splitting the dataset into train and test sets: 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

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
    print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
    print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))






