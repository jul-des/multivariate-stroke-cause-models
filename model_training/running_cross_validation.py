

import pandas as pd
from utils import *

data = pd.read_csv("path/to/data.csv")

base = [] #column index of patient id column
clinical = [] #list of column indices with clinical parameters
treatment = [] #list of column indices with treatment parameters
lesion_features = [] #list of column indices with lesion features


all_results = {"Model": [], "Data": []}
include_treatment = True
for sel_data in ["clinical"]:
    print ("Data:", sel_data)
    if sel_data=="clinical":
        selected_features = base + clinical
    elif sel_data == "lesion_features":
         selected_features = base + lesion_features
    else:
         selected_features = base + clinical + lesion_features

    if include_treatment:
         selected_features = selected_features + treatment

    X = data.iloc[:,selected_features]
    print(X.shape)

    y = data["TOAST"]
  
    for model in ["logreg", "svm", "xgb", "rf"]: #spefify which models to train
        print("Model:", model)
        results = train_model_nested_cv(
            X=X,
            y=y,
            model_type=model,
            feature_method="mrmr", 
            include_treatment=include_treatment,
            toast_classes=[3] #set to [2] to train toast 1 vs toast 2 model
        )

        all_results = save_results(results, all_results, "path/to/output/file.csv", model, sel_data)