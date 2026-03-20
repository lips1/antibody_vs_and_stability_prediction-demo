
from fastapi import FastAPI from pydantic import BaseModel
from app.services.stability_service import predict_stability from app.services.viscosity_service import predict_viscosity
app = FastAPI()
class InputData(BaseModel): sequence: str
@app.post("/predict")
def predict(data: InputData): seq = data.sequence
stability = predict_stability(seq) viscosity = predict_viscosity(seq)
return {
"length": len(seq), "stability": stability, "viscosity": viscosity
}

# stability_service.py (deepStabP integration) import subprocess
import pandas as pd
def predict_stability(seq):
df = pd.DataFrame({"sequence":[seq]}) df.to_csv("input.csv", index=False)
subprocess.run("python deepStabP/predict.py --input input.csv --output out.csv", shell=True) result = pd.read_csv("out.csv")

return float(result["stability"][0])

# viscosity_service.py (DeepViscosity integration) import subprocess
import pandas as pd
def predict_viscosity(seq):
df = pd.DataFrame({"sequence":[seq]}) df.to_csv("input.csv", index=False)
subprocess.run("python DeepViscosity/predict.py --input input.csv --output out.csv", shell=True)
result = pd.read_csv("out.csv") return float(result["viscosity"][0])