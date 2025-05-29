import numpy as np

def load_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path, encoding='latin-1')
    return df

def preprocess_data(df):
    df[df == '?'] = np.nan
    for col in ['workclass', 'occupation', 'native.country']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_categorical_features(df):
    from sklearn import preprocessing
    categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    for feature in categorical:
        le = preprocessing.LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    return df

def scale_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def apply_pca(X, n_components=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_

def train_logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return logreg

def predict(logreg, X_test):
    return logreg.predict(X_test)

def plot_explained_variance(pca):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance with 90% Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()