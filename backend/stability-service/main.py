from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import sys
import os

v1 = None
_model_available = None

def _load_model():
    global v1, _model_available
    if _model_available is not None:
        return
    deepstabp_api = os.environ.get(
        "DEEPSTABP_API_PATH",
        os.path.join(os.path.dirname(__file__), "deepStabP", "src", "Api"),
    )
    if os.path.isdir(deepstabp_api) and deepstabp_api not in sys.path:
        sys.path.insert(0, deepstabp_api)
    try:
        from app.routers import version1
        v1 = version1
        _model_available = True
    except Exception as e:
        print("[stability] model not available:", e)
        _model_available = False

app = FastAPI(title="Stability Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    heavy_chain: str


def _ensure_model_available():
    _load_model()
    if not _model_available:
        raise HTTPException(status_code=503, detail="deepStabP model is not available")


@app.post("/predict")
def predict(data: Input):
    _ensure_model_available()
    try:
        records = [v1.FastaRecord(header="seq1", sequence=" ".join(list(data.heavy_chain.upper())))]
        out = v1.determine_tm(
            records, cell_type="Cell", temperature=37,
            model=v1.model, prediction_net=v1.prediction_net,
            new_features=v1.new_features, tokenizer=v1.tokenizer)
        return {"stability": float(out["Tm"].iloc[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    _ensure_model_available()
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported")

    df = pd.read_csv(file.file)
    cols = set(df.columns)
    if "VH" in cols and "Heavy_Chain" not in cols:
        df = df.rename(columns={"VH": "Heavy_Chain", "VL": "Light_Chain"})

    results = []
    for _, row in df.iterrows():
        try:
            seq = row["Heavy_Chain"]
            records = [v1.FastaRecord(header=row["Name"], sequence=" ".join(list(seq.upper())))]
            out = v1.determine_tm(
                records, cell_type="Cell", temperature=37,
                model=v1.model, prediction_net=v1.prediction_net,
                new_features=v1.new_features, tokenizer=v1.tokenizer)
            results.append({"Name": row["Name"], "Stability": float(out["Tm"].iloc[0])})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"deepStabP failed for row '{row.get('Name', '')}': {e}")

    return Response(
        content=pd.DataFrame(results).to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=stability_results.csv"},
    )
