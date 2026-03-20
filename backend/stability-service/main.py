
from fastapi import FastAPI
from pydantic import BaseModel
from predictor import predictor

app = FastAPI()

class SequenceRequest(BaseModel):
    heavy_chain: str
    light_chain: str = ""

@app.post("/predict")
def predict_stability(req: SequenceRequest):
    try:
        value = predictor.predict(req.heavy_chain, req.light_chain)
        return {"stability": float(value), "status": "success"}
    except Exception as e:
        return {"stability": 0.75, "status": "error", "error": str(e)[:100]}