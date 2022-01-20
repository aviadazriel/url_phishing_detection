import pandas as pd
from features_extractor import UrlFeaturizer
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

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

def calculate_accuracy(y_train, train_conf, y_test, test_conf, model_name):
    # computing the accuracy of the model performance
    acc_train = accuracy_score(y_train, train_conf)
    acc_test  = accuracy_score(y_test, test_conf)
    print("{}: Accuracy on training Data: {:.3f}".format(model_name, acc_train))
    print("{}: Accuracy on test Data: {:.3f}".format(model_name, acc_test))
    return acc_train, acc_test