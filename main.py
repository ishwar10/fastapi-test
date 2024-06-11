# main.py
from fastapi import FastAPI
import pandas as pd
import numpy as np
from joblib import load
import uvicorn
from uvicorn import run as app_run
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import io

app = FastAPI()
model=load("malwareclassifier.joblib")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        # Convert the file contents to a pandas DataFrame
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df=df.drop("class",axis=1)
        # Assuming the model expects a numpy array, convert DataFrame to numpy array
        #input_array = input_df.to_numpy()
        # Make predictions
        predictions = model.predict(df.values)
        # Convert predictions to list if necessary
        predictions_list = predictions.tolist()
        return {"predictions": predictions_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
