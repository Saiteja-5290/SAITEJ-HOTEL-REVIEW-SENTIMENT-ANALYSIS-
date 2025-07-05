# HOTEL REVIEW SENTIMENT ANALYSIS 
import os
import getpass
import pandas as pd
import json
import time
import types
import ibm_boto3
from ibm_botocore.client import Config
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes

print("✅ Libraries loaded.")

# =============================
# CONFIGURATION
# =============================
print("\n🔐 Setting up authentication...")

api_details = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": getpass.getpass("🔑 Enter your IBM Cloud API Key: ")
}

project_id = os.environ.get("PROJECT_ID") or input("🆔 Enter Watsonx Project ID: ")

# Cloud Object Storage (COS) settings
cos_url = 'https://s3.us-south.cloud-object-storage.appdomain.cloud'
cos_key = api_details['apikey']
cos_crn = 'YOUR_COS_INSTANCE_CRN'
bucket = 'bucket-94a3z5f6395wr29'
input_file = 'hotel_reviews (1).csv'

# =============================
# LOAD DATA FROM COS
# =============================
print(f"\n📦 Loading dataset from bucket: {bucket}/{input_file}")

cos = ibm_boto3.client(
    service_name='s3',
    ibm_api_key_id=cos_key,
    ibm_service_instance_id=cos_crn,
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=cos_url
)

def __iter__(self): return 0

try:
    obj = cos.get_object(Bucket=bucket, Key=input_file)['Body']
    if not hasattr(obj, '__iter__'):
        obj.__iter__ = types.MethodType(__iter__, obj)
    reviews_df = pd.read_csv(obj)
    print("✅ Data loaded:")
    print(reviews_df.head())
except Exception as ex:
    print("❌ Failed to load data:", ex)
    reviews_df = None

# =============================
# SETTING UP FOUNDATION MODEL
# =============================
if reviews_df is not None:
    print("\n🧠 Initializing foundation model...")

    config_params = {
        GenParams.MAX_NEW_TOKENS: 150,
        GenParams.MIN_NEW_TOKENS: 10,
        GenParams.TEMPERATURE: 0.1,
    }

    prompt_template = '''
Classify the review sentiment as Positive, Negative, or Neutral.
Also identify key topics from this list only:
["Room Cleanliness", "Staff Friendliness", "Check-in/Check-out", "Food Quality", "Room Service", "Location", "Amenities", "Value for Money"]
Return the result in a valid JSON format.
'''

    llm = Model(
        model_id=ModelTypes.FLAN_UL2,
        params=config_params,
        credentials=api_details,
        project_id=project_id
    )
    print("✅ Model is ready.")

# =============================
# RUN ANALYSIS ON EACH REVIEW
# =============================
    print("\n🔍 Analyzing reviews...")
    final_outputs = []
    sentiment_labels = []
    topic_labels = []

    for idx, row in enumerate(reviews_df.review_text):
        print(f"\n📄 Review {idx + 1}/{len(reviews_df)}")
        query = f"{prompt_template}\n\nReview: {row}\nResult:"

        try:
            raw = llm.generate_text(prompt=query).strip()
            print(f"LLM Response: {raw}")

            clean_json = raw if raw.startswith("{") else "{" + raw.strip(',') + "}"
            parsed = json.loads(clean_json)

            sentiment_labels.append(parsed.get("sentiment", "Unknown"))
            topic_labels.append(', '.join(parsed.get("topics", [])))
            final_outputs.append(raw)
        
        except Exception as err:
            print(f"⚠️ Error at review {idx+1}: {err}")
            sentiment_labels.append("Parse Error")
            topic_labels.append("Could not extract")
            final_outputs.append("")

        time.sleep(0.4)

    reviews_df['Sentiment'] = sentiment_labels
    reviews_df['Service_Topics'] = topic_labels
    print("\n✅ Completed sentiment analysis.")

# =============================
# SAVE FINAL OUTPUT
# =============================
    output_path = 'sentiment_analysis_result.csv'
    reviews_df.to_csv(output_path, index=False)
    print(f"\n📁 Final report saved as: {output_path}")
