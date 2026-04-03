import os
import subprocess
import sys
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="DeepViscosity API")

# Path to your cloned repo
default_deeppath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "DeepViscosity")
)
DEEPPATH = os.environ.get("DEEPPATH", default_deeppath)

# Check predictor 
_predictor_ready = False
try:
    import joblib  # noqa: F401
    import tensorflow  # noqa: F401
    import anarci  # noqa: F401
    _predictor_ready = True
except ImportError:
    print("[viscosity] Some ML deps missing")

# -----------------------------
# REQUEST MODEL
# -----------------------------
class SequenceRequest(BaseModel):
    name: str = "sample_1"
    heavy_chain: str
    light_chain: str


# -----------------------------
# RUN MODEL FROM CLONEPROJECT
# -----------------------------
def run_deepviscosity(df) -> "pd.DataFrame":
    try:
        if not _predictor_ready:
            raise RuntimeError("DeepViscosity dependencies are missing")
        # DeepViscosity predictor expects this filename and column names
        input_file = os.path.join(DEEPPATH, "DeepViscosity_input.csv")
        output_file = os.path.join(DEEPPATH, "DeepViscosity_classes.csv")

        # Remove old output
        if os.path.exists(output_file):
            os.remove(output_file)

        # Ensure predictor column names: Heavy_Chain, Light_Chain
        df_to_write = df.copy()
        if "VH" in df_to_write.columns and "VL" in df_to_write.columns:
            df_to_write = df_to_write.rename(columns={"VH": "Heavy_Chain", "VL": "Light_Chain"})

      
        df_to_write.to_csv(input_file, index=False)

        # print("=== INPUT FILE ===")
        # print(open(input_file).read())

       
        # import_check = subprocess.run(
        #     [sys.executable, "-c", "import joblib, tensorflow, anarci"],
        #     capture_output=True,
        #     text=True
        # )
        # if import_check.returncode != 0:
        #     raise RuntimeError("DeepViscosity runtime dependencies check failed")

        #  run original  
        result = subprocess.run(
            [sys.executable, "deepviscosity_predictor.py"],
            cwd=DEEPPATH,
            capture_output=True,
            text=True
        )

        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"DeepViscosity predictor failed: {result.stderr.strip() or result.stdout.strip()}")

        if not os.path.exists(output_file):
            raise RuntimeError("DeepViscosity output file was not generated")

        result_df = pd.read_csv(output_file)

        if result_df.empty:
            raise RuntimeError("DeepViscosity returned an empty result")

        return result_df

    except Exception as e:
        raise RuntimeError(f"DeepViscosity failed: {str(e)}")


# -----------------------------
# SINGLE SEQUENCE API
# -----------------------------
@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        
        df = pd.DataFrame([{
            "Name": req.name,
            "VH": req.heavy_chain,
            "VL": req.light_chain
        }])

       
        result_df = run_deepviscosity(df)

        return {
            "name": result_df.iloc[0]["Name"],
            "viscosity_class": int(result_df.iloc[0]["DeepViscosity_classes"]),
            "prob_mean": float(result_df.iloc[0]["Prob_Mean"]),
            "prob_std": float(result_df.iloc[0]["Prob_Std"]),
            "status": "success"
        }

    except Exception as e:
        # If model/runtime error, surface as 503 (upstream/service unavailable)
        msg = str(e)
        if isinstance(e, RuntimeError) or msg.startswith("DeepViscosity failed") or "missing dependencies" in msg.lower():
            raise HTTPException(status_code=503, detail=msg)
        raise HTTPException(status_code=500, detail=msg)


# -----------------------------
# CSV UPLOAD API
# -----------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload CSV only")

    try:
        # 
        df = pd.read_csv(file.file)
        # Normalize column names: strip whitespace, remove BOM, unify case
        try:
            new_cols = []
            for c in df.columns:
                if isinstance(c, str):
                    nc = c.strip().replace('\ufeff', '')
                else:
                    nc = c
                new_cols.append(nc)
            df.columns = new_cols
        except Exception:
            pass

        # Log incoming CSV for debugging
        try:
            print("[viscosity] received CSV columns:", df.columns.tolist())
            print("[viscosity] preview:\n", df.head().to_dict())
        except Exception:
            pass

        # validation
        cols = {c for c in df.columns}
        lower_map = {str(c).lower(): c for c in df.columns}
        has_vhvl = {"name", "vh", "vl"}.issubset(set(k.lower() for k in df.columns))
        has_heavylight = {"name", "heavy_chain", "light_chain"}.issubset(set(k.lower() for k in df.columns))
        # If VH/VL present with different casing, rename them
        if "vh" in lower_map and "vl" in lower_map:
            vh_col = lower_map.get("vh")
            vl_col = lower_map.get("vl")
            if vh_col != "VH" or vl_col != "VL":
                df = df.rename(columns={vh_col: "VH", vl_col: "VL"})
        if "heavy_chain" in lower_map and "light_chain" in lower_map:
            hc = lower_map.get("heavy_chain")
            lc = lower_map.get("light_chain")
            if hc != "Heavy_Chain" or lc != "Light_Chain":
                df = df.rename(columns={hc: "Heavy_Chain", lc: "Light_Chain"})
        cols = set(df.columns)
        has_vhvl = {"Name", "VH", "VL"}.issubset(cols)
        has_heavylight = {"Name", "Heavy_Chain", "Light_Chain"}.issubset(cols)
        if not (has_vhvl or has_heavylight):
            raise HTTPException(
                status_code=400,
                detail="CSV must include Name and either VH/VL or Heavy_Chain/Light_Chain columns"
            )

        result_df = run_deepviscosity(df)

        return Response(
            content=result_df.to_csv(index=False),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=viscosity_output.csv"
            }
        )

    except Exception as e:
        msg = str(e)
        if isinstance(e, RuntimeError) or msg.startswith("DeepViscosity failed") or "missing dependencies" in msg.lower():
            raise HTTPException(status_code=503, detail=msg)
        raise HTTPException(status_code=500, detail=msg) 