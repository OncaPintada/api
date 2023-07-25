from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = FastAPI()
model_RandomForest = load('./models/randomForest_Oversampling_FeatureSelection.joblib') 
class Item(BaseModel):
    index: list
    v7: list
    v16: list
    v3: list
    v11: list
    v17: list
    v12: list
    v4: list
    v10: list
    v14: list

class Items(BaseModel):
    listItems: object

@app.put("/classificationSingle")
def update_item(item: Item):
    df = item.model_dump()
    df = pd.DataFrame.from_dict(df)
    df = df.iloc[:,1:]
    classification = model_RandomForest.predict(df)
    print(type(classification))
    classification = classification.tolist()
    return {"classification":classification}


@app.put("/classificationAll")
def update_item(items: Items):
    df = items.model_dump()
    df = pd.DataFrame(df['listItems'])
    df = df.transpose()
    df = df.iloc[:,1:]
    classification = model_RandomForest.predict(df)
    classification = classification.tolist()
    return {"classification":classification}