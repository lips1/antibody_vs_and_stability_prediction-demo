from fastapi import FastAPI
from pydantic import BaseModel
from predictor import predictor

app = FastAPI()

class SequenceRequest(BaseModel):
    heavy_chain: str
    light_chain: str = ""

@app.post("/predict")
def predict_viscosity(req: SequenceRequest):
    """Predict viscosity from sequences using git DeepViscosity models"""
    try:
        value = predictor.predict(req.heavy_chain, req.light_chain)
        return {"viscosity": float(value), "status": "success"}
    except Exception as e:
        # Fallback on any error
        return {"viscosity": 20.0, "status": "error", "error": str(e)[:100]}
