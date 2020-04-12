'''
April 2020
Xinru Yan
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from preprocess import Conversation
from preprocess import read_pickle
import joblib
import pandas as pd
import config


def tfidf_features(x):
    tfidf_vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 3), stop_words='english')
    features = tfidf_vectorizer.fit_transform(x)

    return features


def create_features(i, df):
    assert 1 <= i <= 4, 'only three feature options'
    if i == 1:
        features = df[['B4', 'New Label', 'Unit']]
    elif i == 2:
        features = df[['New Label', 'Unit']]
    elif i == 3:
        features = df[['B4', 'Unit']]
    else:
        features = tfidf_features(df['Unit'])
    return features


def create_pipeline(i, model_option):
    assert 1 <= i <= 4, 'only three feature options'
    if i != 4:
        if i == 1:
            preprocess = ColumnTransformer([('speaker', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['B4']),
                                          ('semantic_feature', OneHotEncoder(dtype='int', handle_unknown='ignore'),['New Label']),
                                          ('text_tfidf', TfidfVectorizer(min_df=5, ngram_range=(1, 3), stop_words='english'),'Unit')],
                                         remainder='passthrough')
        elif i == 2:
            preprocess = ColumnTransformer([('semantic_feature', OneHotEncoder(dtype='int', handle_unknown='ignore'),['New Label']),
                                            ('text_tfidf',TfidfVectorizer(min_df=5, ngram_range=(1, 3), stop_words='english'), 'Unit')],
                                           remainder='passthrough')
        elif i == 3:
            preprocess = ColumnTransformer([('speaker', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['B4']),
                                            ('text_tfidf',TfidfVectorizer(min_df=5, ngram_range=(1, 3), stop_words='english'), 'Unit')],
                                           remainder='passthrough')
        if model_option == 'SVM':
            model = make_pipeline(preprocess, LinearSVC())
        elif model_option == 'LR':
            model = make_pipeline(preprocess, LogisticRegression(random_state=0, max_iter=200))
        else:
            model = make_pipeline(preprocess, MultinomialNB())
    else:
        if model_option == 'SVM':
            model = LinearSVC()
        elif model_option == 'LR':
            model = LogisticRegression(random_state=0, max_iter=200)
        else:
            model = MultinomialNB()
    return model


def train_and_test_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred, zero_division=0))


def save_model(model, feature_option, model_option):
    joblib.dump(model, config.model_path + str(model.__class__.__name__) + '_' + str(feature_option) + '_' + model_option + '.pkl')


def main():
    df = pd.read_csv('../data/raw/semantic.csv')
    df["New Label"] = df["New Label"].fillna("none")
    df["Poss_labels"] = df["Poss_labels"].fillna("none")
    df["New Label"] = df["New Label"].astype("category")
    df["Poss_labels"] = df["Poss_labels"].astype("category")

    test_convos = read_pickle('../data/preprocessed/102_test_prepared_data.pickle')

    test_dataset = df.loc[df['B2'].isin(list(test_convos.keys()))]
    train_dataset = df[~df.index.isin(test_dataset.index)]


    X_train_1, X_test_1, y_train_1, y_test_1 = create_features(1, train_dataset), create_features(1, test_dataset), train_dataset.Poss_labels.values, test_dataset.Poss_labels.values
    X_train_2, X_test_2, y_train_2, y_test_2 = create_features(2, train_dataset), create_features(2, test_dataset), train_dataset.Poss_labels.values, test_dataset.Poss_labels.values
    X_train_3, X_test_3, y_train_3, y_test_3 = create_features(3, train_dataset), create_features(3, test_dataset), train_dataset.Poss_labels.values, test_dataset.Poss_labels.values
    X_train, X_test, y_train, y_test = train_test_split(create_features(4, df), df.Poss_labels.values, test_size=0.0935, random_state=0)

    models = ['SVM', 'LR', 'NB']

    for model in models:
        print(f'training and testing {model} model')

        print(f'speaker + semantic + text')
        combined_model_1 = create_pipeline(1, model)
        train_and_test_model(combined_model_1, X_train_1, X_test_1, y_train_1, y_test_1)
        save_model(combined_model_1, 1, model)

        print(f'semantic + text')
        combined_model_2 = create_pipeline(2, model)
        train_and_test_model(combined_model_2, X_train_2, X_test_2, y_train_2, y_test_2)
        save_model(combined_model_2, 2, model)

        print(f'speaker + text')
        combined_model_3 = create_pipeline(3, model)
        train_and_test_model(combined_model_3, X_train_3, X_test_3, y_train_3, y_test_3)
        save_model(combined_model_3, 3, model)

        print(f'text')
        text_model = create_pipeline(4, model)
        train_and_test_model(text_model, X_train, X_test, y_train, y_test)
        save_model(text_model, 4, model)


if __name__ == '__main__':
    main()

