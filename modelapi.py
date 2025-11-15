from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
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
            raise FileNotFoundError("sectorwise_multipliers.csv not found")
        
        # Load project data from both CSV files
        project_path1 = Path("project_data.csv")
        project_path2 = Path("project_data1.csv")
        
        project_data_parts = []
        
        if project_path1.exists():
            df1 = pd.read_csv(project_path1)
            project_data_parts.append(df1)
            logger.info(f"Loaded project_data.csv with {len(df1)} rows")
        else:
            raise FileNotFoundError("project_data.csv not found")
            
        if project_path2.exists():
            df2 = pd.read_csv(project_path2)
            project_data_parts.append(df2)
            logger.info(f"Loaded project_data1.csv with {len(df2)} rows")
        else:
            logger.warning("project_data1.csv not found, using only project_data.csv")
        
        # Combine all project data parts
        if project_data_parts:
            PROJECT_DATA = pd.concat(project_data_parts, ignore_index=True)
            logger.info(f"Combined project data with {len(PROJECT_DATA)} total rows")
            
            # Log unique values for debugging
            logger.info(f"Available locations: {PROJECT_DATA['Country'].unique().tolist()}")
            logger.info(f"Available products: {PROJECT_DATA['Main_Prod'].unique().tolist()}")
        else:
            raise FileNotFoundError("No project data files found")
            
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

@app.post("/run_analysis")
async def run_analysis(request: AnalysisRequest):
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
            # Provide more helpful error message with available options
            available_locations = PROJECT_DATA['Country'].unique().tolist()
            available_products = PROJECT_DATA['Main_Prod'].unique().tolist()
            
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"No project data found for location '{request.location}' and product '{request.product}'",
                    "available_locations": available_locations,
                    "available_products": available_products
                }
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
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/available_options")
async def get_available_options():
    """Endpoint to get available locations and products"""
    if PROJECT_DATA is None:
        raise HTTPException(status_code=500, detail="Project data not loaded")
    
    return {
        "locations": PROJECT_DATA['Country'].unique().tolist(),
        "products": PROJECT_DATA['Main_Prod'].unique().tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
