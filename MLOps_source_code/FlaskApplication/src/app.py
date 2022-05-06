import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, request
load_dotenv(Path(".env"))
import pandas as pd
import os
import json

if os.environ.get("ENV", "dev") == "prod":
    load_dotenv(Path(".env.prod"))
if os.environ.get("ENV", "dev") == "dev":
    load_dotenv(Path(".env.dev"))

from logging_module import logger
from predictor import predict
app = Flask(__name__)

@app.route("/health-status")
def get_health_status():
    logger.debug("Health check api version 2")
    resp = jsonify({"status": "I am alive, version 2"})
    resp.status_code = 200
    return resp

@app.route("/churn-prediction", methods=['POST'])
def churn_prediction():
    logger.debug("Churn Prediction API Called")
    df = pd.DataFrame(request.json["data"])
    status, result = predict(df)
    print(result)
    if status == 200:
        result = json.loads(result.to_json(orient="records"))
        resp = jsonify({"result": result})
    else:
        resp = jsonify({"errorDetails": result})
    resp.status_code = status
    return resp
    
    

if __name__ == "__main__":
    app.run(debug=True)
