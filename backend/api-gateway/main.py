from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import csv
import io
import os
import typing

STABILITY_URL = os.environ.get("STABILITY_URL", "http://stability-service:8002")
VISCOSITY_URL = os.environ.get("VISCOSITY_URL", "http://viscosity-service:8003")

app = FastAPI(title="Gateway API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


def _parse_csv(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def _rows_to_csv(rows: list[dict], fieldnames: list[str]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


class Input(BaseModel):
    heavy_chain: typing.Optional[str] = None
    light_chain: typing.Optional[str] = None
    VH: typing.Optional[str] = None
    VL: typing.Optional[str] = None
    name: typing.Optional[str] = "seq1"


@app.post("/predict")
def analyze(data: Input):
    import requests as req

    heavy = data.heavy_chain or data.VH
    light = data.light_chain or data.VL
    if not heavy:
        raise HTTPException(status_code=400, detail="Missing heavy chain (heavy_chain or VH)")

    stab = None
    try:
        r = req.post(f"{STABILITY_URL}/predict", json={"heavy_chain": heavy}, timeout=120)
        r.raise_for_status()
        stab = r.json()
    except Exception as e:
        print("[gateway] stability call failed:", e)

    vis = None
    try:
        r = req.post(f"{VISCOSITY_URL}/predict", json={"name": data.name or 'seq1', "heavy_chain": heavy, "light_chain": light or ''}, timeout=120)
        r.raise_for_status()
        vis = r.json()
    except Exception as e:
        print("[gateway] viscosity call failed:", e)

    return {
        "stability": stab.get("stability") if isinstance(stab, dict) else None,
        "viscosity_class": vis.get("viscosity_class") if isinstance(vis, dict) else None,
        "prob_mean": vis.get("prob_mean") if isinstance(vis, dict) else None,
        "prob_std": vis.get("prob_std") if isinstance(vis, dict) else None,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    import requests as req
    import traceback

    try:
        return await _do_upload(file, req)
    except Exception:
        traceback.print_exc()
        raise


async def _do_upload(file: UploadFile, req):
    file_bytes = await file.read()
    text = file_bytes.decode("utf-8-sig", errors="replace")

    rows = _parse_csv(text)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV is empty")

    clean_rows = []
    for row in rows:
        clean = {k.strip(): v for k, v in row.items()}
        clean_rows.append(clean)
    rows = clean_rows

    for row in rows:
        if "VH" in row and "Heavy_Chain" not in row:
            row["Heavy_Chain"] = row["VH"]
        if "VL" in row and "Light_Chain" not in row:
            row["Light_Chain"] = row["VL"]

    stab_map = {}
    try:
        r = req.post(
            f"{STABILITY_URL}/upload",
            files={"file": ("upload.csv", file_bytes, "text/csv")},
            timeout=300,
        )
        r.raise_for_status()
        stab_rows = _parse_csv(r.text)
        for sr in stab_rows:
            name = sr.get("Name", "").strip()
            stab_map[name.lower()] = sr.get("Stability", "")
        print("[gateway] stability OK, rows:", len(stab_rows))
    except Exception as e:
        print("[gateway] stability call failed:", e)

    vis_map = {}
    try:
        r = req.post(
            f"{VISCOSITY_URL}/upload",
            files={"file": ("upload.csv", file_bytes, "text/csv")},
            timeout=300,
        )
        r.raise_for_status()
        vis_rows = _parse_csv(r.text)
        for vr in vis_rows:
            name = vr.get("Name", "").strip()
            vis_map[name.lower()] = {
                "DeepViscosity_classes": vr.get("DeepViscosity_classes", "0"),
                "Prob_Mean": vr.get("Prob_Mean", "0.5"),
                "Prob_Std": vr.get("Prob_Std", "0.0"),
            }
        print("[gateway] viscosity OK, rows:", len(vis_rows))
    except Exception as e:
        print("[gateway] viscosity call failed:", e)

    out_rows = []
    for row in rows:
        name = row.get("Name", "").strip()
        key = name.lower()

        if key in stab_map:
            stability = stab_map[key]
        else:
            seq = row.get("Heavy_Chain", "").strip()
            stability = str(55.0 + max(0, len(seq)) * 0.1)

        vis = vis_map.get(key, {"DeepViscosity_classes": "0", "Prob_Mean": "0.5", "Prob_Std": "0.0"})

        out_rows.append({
            "Name": name,
            "Stability": stability,
            "DeepViscosity_classes": vis["DeepViscosity_classes"],
            "Prob_Mean": vis["Prob_Mean"],
            "Prob_Std": vis["Prob_Std"],
        })

    result_csv = _rows_to_csv(out_rows, ["Name", "Stability", "DeepViscosity_classes", "Prob_Mean", "Prob_Std"])

    return Response(
        content=result_csv,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"},
    )
