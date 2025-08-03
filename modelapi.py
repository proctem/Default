from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import numpy as np
import logging
from originalmodel import Analytics_Model  # Make sure this import is correct

# ---------------------------------------
# Set up error logging to error.log file
# ---------------------------------------
logging.basicConfig(
    filename='error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------
# Initialize FastAPI
# ---------------------------------------
app = FastAPI(title="Integrated Project Economics Model API")

# ---------------------------------------
# Global exception handler
# ---------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "An internal error occurred. Please check the logs."}
    )

# ---------------------------------------
# Load static data (only once at startup)
# ---------------------------------------
try:
    project_datas = pd.read_csv("./project_data.csv")
    multipliers = pd.read_csv("./sectorwise_multipliers.csv")
except Exception as e:
    logger.error(f"Error loading CSV files: {e}", exc_info=True)
    raise RuntimeError("Required CSV files could not be loaded. Check file paths or formats.")

# ---------------------------------------
# Define request schema
# ---------------------------------------
class ModelInput(BaseModel):
    location: str
    product: str
    plant_mode: Literal["Green", "Brown"]
    fund_mode: Literal["Debt", "Equity"]
    opex_mode: Literal["Constant", "Inflated"]
    carbon_value: Literal["Yes", "No"]

# ---------------------------------------
# POST endpoint to run the model
# ---------------------------------------
@app.post("/run_model/")
def run_model(input_data: ModelInput):
    try:
        results = Analytics_Model(
            multiplier=multipliers,
            project_data=project_datas,
            location=input_data.location,
            product=input_data.product,
            plant_mode=input_data.plant_mode,
            fund_mode=input_data.fund_mode,
            opex_mode=input_data.opex_mode,
            carbon_value=input_data.carbon_value
        )
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Error during model execution: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
