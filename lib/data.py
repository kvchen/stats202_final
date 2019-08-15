from .metadata import COLUMN_MEANINGS

import category_encoders as ce
import pandas as pd

from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def normalize_data(df):
    x_scaled = MinMaxScaler().fit_transform(df.values)
    return pd.DataFrame(x_scaled, columns=df.columns, index=df.index)


def parse_data(data_dir, glob):
    df = pd.concat([
        pd.read_csv(filename, index_col="AssessmentiD")
        for filename in Path(data_dir).glob(glob)
    ])
    return df
    #return df.rename(columns=COLUMN_MEANINGS)


def read_study_data(study):
    return parse_data('data/', f"Study_{study}.csv")


def read_train_data():
    return parse_data('data/', 'Study_[ABCD].csv')


def read_test_data():
    return parse_data('data/', 'Study_E.csv')


def split_data(df):
    y = (df['LeadStatus'] != 'Passed') * 1
    X = df.drop(['LeadStatus'], axis=1)
    
    return X, y


# def featurize(X):
#     X = X.drop(['Study', 'PatientID'], axis=1)
#     group_cols = ['Country', 'SiteID', 'RaterID']
#     zscore = lambda x: (x - x.mean()) / x.std()
    
#     # Standardize each group
#     X = X.groupby(group_cols).transform(zscore)
    
#     one_hot = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True, return_df=True)
#     X = one_hot.fit_transform(X)
    
#     return X


def group_transform(g):
    g['PANSS_TotalDelta'] = g.PANSS_Total - g.PANSS_Total.iloc[0]
    
    baseline = g[g.VisitDay == 0].mean().to_frame().T
    baseline = baseline.filter(regex="((P|N|G)\d+|Control|Treatment)").add_prefix('baseline_')

    baseline_rep = pd.DataFrame(
        baseline.values.repeat(len(g), axis=0),
        columns=baseline.columns,
        index=g.index
    )

    return g.join(baseline_rep)


def augment_data(df):
    df = df.copy()
    
    df['PositiveScore'] = df.filter(regex="P\d+").sum(axis=1)
    df['NegativeScore'] = df.filter(regex="N\d+").sum(axis=1)
    df['GeneralScore'] = df.filter(regex="G\d+").sum(axis=1)
    df['CompositeScore'] = df.PositiveScore - df.NegativeScore
    df['Control'] = df.TxGroup == 'Control'
    df['Treatment'] = df.TxGroup == 'Treatment'

    return df.groupby('PatientID').apply(group_transform)
    

def get_resampled_dataset(X, y):
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled)
    
    return X_resampled, y_resampled


def get_classification_datasets():
    train_df = augment_data(read_train_data())
    X_test = augment_data(read_test_data())
    X_train, y_train = split_data(train_df)

    #     X_concat = pd.concat([X_train_raw, X_test_raw])
    #     X_train_featurized = featurize(X_concat)

    #     X_train = X_train_featurized[:len(X_train_raw)]
    #     X_test = X_train_featurized[len(X_train_raw):]
    
    return X_train, y_train, X_test
