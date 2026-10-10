from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Pro Urdu AI Backend is Running! Go to /docs to test the model."}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Production NLP models (26k dataset)...")
model = joblib.load('production_urdu_model.pkl')
vectorizer = joblib.load('production_urdu_vectorizer.pkl')
print("Production models loaded successfully!")

class TextRequest(BaseModel):
    text: str

@app.post("/api/analyze")
async def analyze_sentiment(request: TextRequest):
    vectorized_text = vectorizer.transform([request.text])
    prediction = model.predict(vectorized_text)[0]
    
    return {
        "text": request.text,
        "sentiment": prediction  # Returns "positive" or "negative" directly
    }