
import evalml
import woodwork as ww
import pandas as pd

PATH_TO_TRAIN = "train"
PATH_TO_HOLDOUT = "holdout"
TARGET = "f_0"
column_mapping = "columnMapping.json"

# This is the machine learning pipeline you have exported.
# By running this code you will fit the pipeline on the files provided
# and you can then use this pipeline for prediction and model understanding.
from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline
from featuretools import load_features

features = load_features("features.json")
pipeline = BinaryClassificationPipeline(
    component_graph={
        "DFS Transformer": ["DFS Transformer", "X", "y"],
        "Label Encoder": ["Label Encoder", "X", "y"],
        "Numeric Pipeline - Select Columns By Type Transformer": [
            "Select Columns By Type Transformer",
            "DFS Transformer.x",
            "Label Encoder.y",
        ],
        "Numeric Pipeline - Label Encoder": [
            "Label Encoder",
            "Numeric Pipeline - Select Columns By Type Transformer.x",
            "Label Encoder.y",
        ],
        "Numeric Pipeline - Imputer": [
            "Imputer",
            "Numeric Pipeline - Select Columns By Type Transformer.x",
            "Numeric Pipeline - Label Encoder.y",
        ],
        "Numeric Pipeline - Standard Scaler": [
            "Standard Scaler",
            "Numeric Pipeline - Imputer.x",
            "Numeric Pipeline - Label Encoder.y",
        ],
        "Numeric Pipeline - Select Columns Transformer": [
            "Select Columns Transformer",
            "Numeric Pipeline - Standard Scaler.x",
            "Numeric Pipeline - Label Encoder.y",
        ],
        "Categorical Pipeline - Select Columns Transformer": [
            "Select Columns Transformer",
            "DFS Transformer.x",
            "Label Encoder.y",
        ],
        "Categorical Pipeline - Label Encoder": [
            "Label Encoder",
            "Categorical Pipeline - Select Columns Transformer.x",
            "Label Encoder.y",
        ],
        "Categorical Pipeline - Imputer": [
            "Imputer",
            "Categorical Pipeline - Select Columns Transformer.x",
            "Categorical Pipeline - Label Encoder.y",
        ],
        "Categorical Pipeline - One Hot Encoder": [
            "One Hot Encoder",
            "Categorical Pipeline - Imputer.x",
            "Categorical Pipeline - Label Encoder.y",
        ],
        "Categorical Pipeline - Standard Scaler": [
            "Standard Scaler",
            "Categorical Pipeline - One Hot Encoder.x",
            "Categorical Pipeline - Label Encoder.y",
        ],
        "Oversampler": [
            "Oversampler",
            "Numeric Pipeline - Select Columns Transformer.x",
            "Categorical Pipeline - Standard Scaler.x",
            "Categorical Pipeline - Label Encoder.y",
        ],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Oversampler.x",
            "Oversampler.y",
        ],
    },
    parameters={
        "DFS Transformer": {"features": features},
        "Label Encoder": {"positive_label": None},
        "Numeric Pipeline - Select Columns By Type Transformer": {
            "column_types": ["category", "EmailAddress", "URL"],
            "exclude": True,
        },
        "Numeric Pipeline - Label Encoder": {"positive_label": None},
        "Numeric Pipeline - Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
            "boolean_impute_strategy": "most_frequent",
            "categorical_fill_value": None,
            "numeric_fill_value": None,
            "boolean_fill_value": None,
        },
        "Numeric Pipeline - Select Columns Transformer": {
            "columns": [
                "f_25",
                "f_18",
                "f_22",
                "f_24",
                "f_13",
                "f_14",
                "f_19",
                "f_15",
                "f_12",
                "f_10",
                "f_21",
                "f_8",
                "f_3",
                "f_7",
                "f_20",
                "f_16",
                "f_6",
                "f_26",
                "f_23",
                "f_4",
            ]
        },
        "Categorical Pipeline - Select Columns Transformer": {
            "columns": ["f_11", "f_9", "f_1"]
        },
        "Categorical Pipeline - Label Encoder": {"positive_label": None},
        "Categorical Pipeline - Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
            "boolean_impute_strategy": "most_frequent",
            "categorical_fill_value": None,
            "numeric_fill_value": None,
            "boolean_fill_value": None,
        },
        "Categorical Pipeline - One Hot Encoder": {
            "top_n": 10,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
        "Oversampler": {
            "sampling_ratio": 0.25,
            "k_neighbors_default": 5,
            "n_jobs": -1,
            "sampling_ratio_dict": None,
            "categorical_features": [
                7,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
            ],
            "k_neighbors": 5,
        },
        "Logistic Regression Classifier": {
            "penalty": "l2",
            "C": 1.0,
            "n_jobs": -1,
            "multi_class": "auto",
            "solver": "lbfgs",
        },
    },
    random_seed=0,
)


print(pipeline.name)
print(pipeline.parameters)
pipeline.describe()

df = ww.deserialize.from_disk(PATH_TO_TRAIN)
y_train = df.ww[TARGET]
X_train = df.ww.drop(TARGET)
pipeline.fit(X_train, y_train)

# You can now generate predictions as well as run model understanding.
df = ww.deserialize.from_disk(PATH_TO_HOLDOUT)
y_holdout = df.ww[TARGET]
X_holdout = df.ww.drop(TARGET)

pipeline.predict(X_holdout)

# Note: if you have a column mapping, to predict on new data you have on hand
# Map the column names and run prediction
# X_test = X_test.rename(column_mapping, axis=1)
# pipeline.predict(X_test)

# For more info please check out:
# https://evalml.alteryx.com/en/stable/user_guide/automl.html
