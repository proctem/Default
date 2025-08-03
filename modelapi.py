from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import originalmodela as model
import pandas as pd
import numpy as np
import logging
from copy import deepcopy
from pathlib import Path
import json
import traceback
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Store pristine defaults at startup
DEFAULT_PARAMS = None
MULTIPLIER_DATA = None
PROJECT_DATA = None

def load_data_files():
    """Load all required data files at startup"""
    global DEFAULT_PARAMS, MULTIPLIER_DATA, PROJECT_DATA
    
    try:
        # Load default parameters
        DEFAULT_PARAMS = deepcopy(model.PARAMS)
        logger.info("Loaded default parameters")
        
        # Load multiplier data
        multiplier_path = Path("sectorwise_multipliers.csv")
        if multiplier_path.exists():
            MULTIPLIER_DATA = pd.read_csv(multiplier_path)
            logger.info("Loaded multiplier data")
        else:
            raise FileNotFoundError("multiplier_data.csv not found")
        
        # Load project data
        project_path = Path("project_data.csv")
        if project_path.exists():
            PROJECT_DATA = pd.read_csv(project_path)
            logger.info("Loaded project data")
        else:
            raise FileNotFoundError("project_data.csv not found")
            
    except Exception as e:
        logger.critical(f"Failed to load data files: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize application data"""
    try:
        load_data_files()
        logger.info("Application startup completed")
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        raise RuntimeError("Application failed to initialize")

@contextmanager
def request_context():
    """Context manager to handle request-specific state"""
    try:
        # Reset to defaults at start of each request
        model.PARAMS = deepcopy(DEFAULT_PARAMS)
        logger.debug("Reset PARAMS to defaults")
        
        # Create fresh copies of data
        multiplier_data = MULTIPLIER_DATA.copy()
        project_data = PROJECT_DATA.copy()
        logger.debug("Created fresh data copies")
        
        yield {
            "PARAMS": model.PARAMS,
            "multiplier_data": multiplier_data,
            "project_data": project_data
        }
        
    except Exception as e:
        logger.error(f"Request context setup failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")


class AnalysisRequest(BaseModel):
    # Required parameters
    location: str
    product: str
    plant_mode: str  # "Green" or "Brown"
    fund_mode: str   # "Debt", "Equity", or "Mixed"
    
    # Optional parameters with defaults
    opex_mode: Optional[str] = "Inflated"
    plant_size: Optional[str] = "Large"
    plant_effy: Optional[str] = "High"
    carbon_value: Optional[str] = "No"
    

@app.on_event("startup")
async def startup_event():
    """Load data files when starting the application"""
    load_data_files()
# Add this to your API code (after loading DEFAULT_PARAMS)
DEFAULT_PARAMS = deepcopy(model.PARAMS)  # Store pristine defaults at startup

@app.post("/run_analysis")
async def run_analysis(request: AnalysisRequest):
    print(f"Current PARAMS at start: {model.PARAMS['construction_prd']}")  # Debug log
    try:
        # Validate we have the required data
        if MULTIPLIER_DATA is None or PROJECT_DATA is None:
            raise HTTPException(status_code=500, detail="Data files not loaded")
        
       
        # Filter project data for this request
        project_data = PROJECT_DATA[
            (PROJECT_DATA['Country'] == request.location) & 
            (PROJECT_DATA['Main_Prod'] == request.product)
        ]
        
        if len(project_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No project data found for location '{request.location}' and product '{request.product}'"
            )

        # Run the analysis
        results = model.Analytics_Model2(
            multiplier=MULTIPLIER_DATA,
            project_data=project_data,
            location=request.location,
            product=request.product,
            plant_mode=request.plant_mode,
            fund_mode=request.fund_mode,
            opex_mode=request.opex_mode,
            carbon_value=request.carbon_value,
            plant_size=request.plant_size,
            plant_effy=request.plant_effy
        )
        
        # Convert results to list of dicts for JSON response
        return results.to_dict(orient='records')

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
