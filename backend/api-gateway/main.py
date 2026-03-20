from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STABILITY_URL = "http://stability-service:8002/predict"
VISCOSITY_URL = "http://viscosity-service:8003/predict"

class SequenceRequest(BaseModel):
    heavy_chain: str
    light_chain: str = ""

@app.post("/analyze")
def analyze(req: SequenceRequest):
    heavy = req.heavy_chain.upper().strip()
    light = req.light_chain.upper().strip() if req.light_chain else ""

    if not heavy:
        return {"error": "Heavy chain is required"}

    # Call stability service
    stability_res = requests.post(
        STABILITY_URL,
        json={"heavy_chain": heavy, "light_chain": light}
    ).json()

    # Call viscosity service
    viscosity_res = requests.post(
        VISCOSITY_URL,
        json={"heavy_chain": heavy, "light_chain": light}
    ).json()

    stability = stability_res.get("stability")
    viscosity = viscosity_res.get("viscosity")

    return {
        "heavy_chain": heavy,
        "light_chain": light,
        "stability": stability,
        "stability_level": "Stable" if stability and stability > 0.7 else "Unstable",
        "viscosity": viscosity,
        "status": "success" if stability is not None and viscosity is not None else "error"
    }