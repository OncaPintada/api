from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_RandomForest = load('./models/randomForest_Oversampling_FeatureSelection2.joblib') 
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
    amount: list

class Items(BaseModel):
    listItems: object

@app.put("/classificationSingle")
def single(item: Item):
    df = item.model_dump()
    df = pd.DataFrame.from_dict(df)
    df = df.iloc[:,1:-1]
    classification = model_RandomForest.predict(df)
    print(type(classification))
    classification = classification.tolist()
    return {"classification":classification}


@app.put("/classificationAll")
def all(items: Items):
    df = items.model_dump()
    df = pd.DataFrame(df['listItems'])
    df = df.transpose()
    df = df.iloc[:,1:-1]
    classification = model_RandomForest.predict(df)
    classification = classification.tolist()
    return {"classification":classification}