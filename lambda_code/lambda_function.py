import json
import boto3
import pickle
import pandas as pd
from io import BytesIO

# S3 model download on cold start
s3 = boto3.client("s3")
BUCKET = "your-s3-bucket-name"
KEY = "fpl_model.pkl"

model_obj = s3.get_object(Bucket=BUCKET, Key=KEY)
model = pickle.load(BytesIO(model_obj["Body"].read()))

def lambda_handler(event, context):
    body = json.loads(event["body"])
    df = pd.DataFrame([body])
    prediction = model.predict(df)[0]

    return {
        "statusCode": 200,
        "body": json.dumps({"expected_points": float(prediction)})
    }
