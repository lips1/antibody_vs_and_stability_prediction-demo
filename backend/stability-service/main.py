# stability_service.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import os
import sys
_api_dir = os.environ.get("DEEPSTABP_API_PATH", "/app/deepStabP/src/Api")
os.chdir(_api_dir)
if _api_dir not in sys.path:
    sys.path.insert(0, _api_dir)

from app.routers import version1 as v1

app = FastAPI(title="Stability Service")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    heavy_chain: str

@app.post("/predict")
def predict(data: Input):
    try:
        records = [v1.FastaRecord(header="seq1", sequence=" ".join(list(data.heavy_chain.upper())))]
        # determine_tm(fasta, lysate, species, transformer, tm_predicter, new_features, tokenizer)
        out = v1.determine_tm(
            records,
            "Cell",
            37,
            v1.model,
            v1.prediction_net,
            v1.new_features,
            v1.tokenizer,
        )
        return {"stability": float(out["Tm"].iloc[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        df = pd.read_csv(file.file)

        required = {"Name", "Heavy_Chain"}
        if not required.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail="CSV must contain Name and Heavy_Chain columns."
            )

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty.")

        records = []
        for _, row in df.iterrows():
            name = str(row["Name"])
            seq = str(row["Heavy_Chain"]).replace(" ", "").upper()
            records.append(
                v1.FastaRecord(
                    header=name,
                    sequence=" ".join(list(seq))
                )
            )

        out = v1.determine_tm(
            records,
            "Cell",
            37,
            v1.model,
            v1.prediction_net,
            v1.new_features,
            v1.tokenizer,
        )

        results = pd.DataFrame({
            "Name": df["Name"].astype(str).values,
            "Stability": out["Tm"].astype(float).values
        })

        merged_csv = results.to_csv(index=False)
        return Response(
            content=merged_csv,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=stability_results.csv"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))