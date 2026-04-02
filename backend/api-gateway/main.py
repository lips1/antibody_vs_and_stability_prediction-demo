
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import pydantic
import requests
import pandas as pd
import io

app = FastAPI(title="Gateway API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Allow frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

class Input(pydantic.BaseModel):
    name: str = "Ab1"
    heavy_chain: str
    light_chain: str


# =====================================================
# SINGLE
# =====================================================
@app.post("/predict")
def analyze(data: Input):

    try:
        # Stability
        stab = requests.post(
            "http://stability-service:8002/predict",
            json={"heavy_chain": data.heavy_chain}
        ).json()

        # Viscosity
        vis = requests.post(
            "http://viscosity-service:8003/predict",
            json={"name": data.name, "heavy_chain": data.heavy_chain, "light_chain": data.light_chain}
        ).json()

        return {
            "stability": stab.get("stability"),
            "viscosity_class": vis.get("viscosity_class"),
            "prob_mean": vis.get("prob_mean"),
            "prob_std": vis.get("prob_std")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# CSV
# =====================================================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    try:
        # Stability CSV
        stab_res = requests.post(
            "http://stability-service:8002/upload",
            files={"file": (file.filename, file.file)}
        )

        stab_df = pd.read_csv(io.StringIO(stab_res.text))

        file.file.seek(0)

        # Viscosity CSV
        vis_res = requests.post(
            "http://viscosity-service:8003/upload",
            files={"file": (file.filename, file.file)}
        )

        vis_df = pd.read_csv(io.StringIO(vis_res.text))

        # Merge
        merged = pd.merge(stab_df, vis_df, on="Name")

        return Response(
            content=merged.to_csv(index=False),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=results.csv"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))