import pandas as pd
from tqdm.notebook import tqdm
from features_extractor import UrlFeaturizer
from sklearn.model_selection import train_test_split

def preprocessing():
    # https://www.phishtank.com/developer_info.php
    df_phishing = pd.read_csv('phising_tank.csv')
    df_phishing["label"] = 1

    # leg urls
    df_leg_link = pd.read_csv('FinalDataset/URL/Benign_list_big_final.csv', names=["url"])
    df_leg_link = df_leg_link.sample(n = len(df_phishing), random_state = 12).copy()
    df_leg_link = df_leg_link.reset_index(drop=True)
    df_leg_link["label"] = 0

    # concat
    df_phis_leg = pd.concat([df_phishing[["url", "label"]], df_leg_link])
    df_phis_leg = df_phis_leg.reset_index()

    # feature extraction
    url_features = UrlFeaturizer("www.google.com")
    features = url_features.run()
    all_features = []
    columns = list(features.keys())
    df = df_phis_leg.copy()
    for i, row in tqdm(df.iterrows()):
        url = row["url"]
        url_features = UrlFeaturizer(url)
        features = url_features.run()
        all_features.append(list(features.values()))

    df[columns] = all_features

    data = df.sample(frac=1).reset_index(drop=True)
    y = data['label']
    X = data.drop(['label', "index", "url", "ext", "hasHttps"], axis=1)
    print( f'X.shape {X.shape}')
    print(f'y.shape {y.shape}')

    # Splitting the dataset into train and test sets: 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # prepare data
    X_train, X_test, y_train, y_test = preprocessing()