import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def clean_df(df):
    df=rmlowstd_col(df, threshold=0.0)
    df=rmsparsityVec(df, sparse_threshold=0.95)
    df=rmCorrCol(df, corrThr=1)

def rmlowstd_col(df,threshold=0.0):
    variance_selector = VarianceThreshold(threshold)
    high_variance_df = pd.DataFrame(variance_selector.fit_transform(df), columns=df.columns[variance_selector.get_support()])
    return high_variance_df

# Filter sparse features
def rmsparsityVec(df,sparse_threshold=0.95):
# rm vectors with more than 95% of zero
    sparse_features = []
    for column in df.columns:
        sparsity = (df[column] == 0).mean()
        if sparsity < sparse_threshold:
            sparse_features.append(column)
    sparse_df = df[sparse_features]
    return sparse_df

def rmCorrCol(df,corrThr=1):
# Filter repetitive features
    correlated_features = set()
    correlation_matrix = df.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= corrThr:  # Repetitive features have a correlation of 1
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    filtered_df = df.drop(correlated_features, axis=1)
    return df

# Function to identify categorical columns
def identify_categorical_columns(df, min_unique=2, max_unique=10):
    categorical_columns = []
    for col in df.columns:
        unique_values = df[col].unique()
        if min_unique <= len(unique_values) <= max_unique:
            categorical_columns.append(col)
    return categorical_columns
#patients with Complete response!!!
#df[df['Initial Treatment Response']=='Complete response']
