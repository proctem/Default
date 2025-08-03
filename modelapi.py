from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from originalmodela import Analytics_Model2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Project Economics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProjectInput(BaseModel):
    location: Optional[str] = "CAN"
    plant_mode: Optional[str] = "Green"
    fund_mode: Optional[str] = "Equity"
    opex_mode: Optional[str] = "Uninflated"
    carbon_value: Optional[str] = "No"
    operating_prd: Optional[int] = 27
    util_operating_first: Optional[float] = 0.70
    util_operating_second: Optional[float] = 0.80
    util_operating_third: Optional[float] = 0.95
    infl: Optional[float] = 0.02
    RR: Optional[float] = 0.035
    IRR: Optional[float] = 0.10
    construction_prd: Optional[int] = 3
    capex_spread: Optional[List[float]] = [0.2, 0.5, 0.3]
    shrDebt_value: Optional[float] = 0.60
    baseYear: Optional[int] = 2025  # Will map to Base_Yr
    ownerCost: Optional[float] = 0.10
    corpTAX: Optional[float] = 0.27
    Feed_Price: Optional[float] = 150.0
    Fuel_Price: Optional[float] = 3.5
    Elect_Price: Optional[float] = 0.12
    CarbonTAX_value: Optional[float] = 56.34  # Maps to CO2price
    credit_value: Optional[float] = 0.10
    CAPEX: Optional[float] = 1080000000
    OPEX: Optional[float] = 33678301.89
    PRIcoef: Optional[float] = 0.3
    CONcoef: Optional[float] = 0.7
    EcNatGas: Optional[float] = 53.6
    ngCcontnt: Optional[float] = 50.3
    eEFF: Optional[float] = 0.50
    hEFF: Optional[float] = 0.80
    Cap: Optional[float] = 250000
    Yld: Optional[float] = 0.771
    feedEcontnt: Optional[float] = 48.1
    Heat_req: Optional[float] = 13.1
    Elect_req: Optional[float] = 0.3
    feedCcontnt: Optional[float] = 64
    custom_params: Optional[Dict[str, Any]] = None

def load_multipliers():
    """Load the sectorwise multipliers CSV"""
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(dir_path, "sectorwise_multipliers.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Multipliers file not found at {csv_path}")
            
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded multipliers with {df.shape[0]} rows")
        return df
    except Exception as e:
        logger.error(f"Multiplier loading failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load multiplier data")

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer)):
        return int(obj)
    if isinstance(obj, (np.floating)):
        return float(obj)
    if isinstance(obj, (np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj

def safe_extract_value(data: pd.DataFrame, column: str, default=None):
    """Safely extract a single value from a DataFrame column"""
    try:
        if column in data.columns:
            value = data[column].iloc[0]
            return str(value) if isinstance(value, (bool, np.bool_)) else value
        return default
    except Exception as e:
        logger.warning(f"Error extracting {column}: {str(e)}")
        return default

def create_project_data(input_data: ProjectInput) -> pd.DataFrame:
    """Create project data with proper column mappings for all expected columns"""
    try:
        project_data = {
            'Country': input_data.location,  # Maps to location in model
            'Plant_Mode': input_data.plant_mode,
            'Fund_Mode': input_data.fund_mode,
            'Opex_Mode': input_data.opex_mode,
            'Carbon_Value': input_data.carbon_value,
            'Operating_prd': input_data.operating_prd,
            'Util_operating_first': input_data.util_operating_first,
            'Util_operating_second': input_data.util_operating_second,
            'Util_operating_third': input_data.util_operating_third,
            'Infl': input_data.infl,
            'RR': input_data.RR,
            'IRR': input_data.IRR,
            'Construction_prd': input_data.construction_prd,
            'Capex_spread': input_data.capex_spread,
            'ShrDebt_value': input_data.shrDebt_value,
            'Base_Yr': input_data.baseYear,  # Maps to Base_Yr in MicroEconomic_Model
            'baseYear': input_data.baseYear,  # Keep both for compatibility
            'OwnerCost': input_data.ownerCost,
            'corpTAX': input_data.corpTAX,
            'Feed_Price': input_data.Feed_Price,
            'Fuel_Price': input_data.Fuel_Price,
            'Elect_Price': input_data.Elect_Price,
            'CO2price': input_data.CarbonTAX_value,  # Maps to CO2price in model
            'Credit_value': input_data.credit_value,
            'CAPEX': input_data.CAPEX,
            'OPEX': input_data.OPEX,
            'PRIcoef': input_data.PRIcoef,
            'CONcoef': input_data.CONcoef,
            'EcNatGas': input_data.EcNatGas,
            'ngCcontnt': input_data.ngCcontnt,
            'eEFF': input_data.eEFF,
            'hEFF': input_data.hEFF,
            'Cap': input_data.Cap,
            'Yld': input_data.Yld,
            'feedEcontnt': input_data.feedEcontnt,
            'Heat_req': input_data.Heat_req,
            'Elect_req': input_data.Elect_req,
            'feedCcontnt': input_data.feedCcontnt,
            'Main_Prod': "Ethylene",
            'Plant_Size': "Large",
            'Plant_Effy': "High",
            'ProcTech': "Standard"
        }

        if input_data.custom_params:
            for k, v in input_data.custom_params.items():
                project_data[k] = v

        df = pd.DataFrame([project_data])
        
        # Verify all required columns exist
        required_columns = [
            'Base_Yr', 'Country', 'Plant_Mode', 'Fund_Mode', 
            'Opex_Mode', 'CO2price', 'corpTAX', 'CAPEX'
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        logger.info(f"Created project data with columns: {df.columns.tolist()}")
        return df
        
    except Exception as e:
        logger.error(f"Project data creation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/project-analytics")
async def run_project_analytics(input_data: ProjectInput):
    try:
        logger.info(f"Starting analysis for {input_data.location}")
        
        multipliers = load_multipliers()
        project_data = create_project_data(input_data)
        
        model_params = {
            'multiplier': multipliers,
            'project_data': project_data,
            'location': safe_extract_value(project_data, 'Country'),
            'product': safe_extract_value(project_data, 'Main_Prod', "Ethylene"),
            'plant_mode': safe_extract_value(project_data, 'Plant_Mode'),
            'fund_mode': safe_extract_value(project_data, 'Fund_Mode'),
            'plant_size': safe_extract_value(project_data, 'Plant_Size', "Large"),
            'plant_effy': safe_extract_value(project_data, 'Plant_Effy', "High"),
            'opex_mode': safe_extract_value(project_data, 'Opex_Mode'),
            'carbon_value': safe_extract_value(project_data, 'Carbon_Value')
        }
        
        # Execute analytics model with error wrapping
        try:
            results = Analytics_Model2(**model_params)
        except KeyError as e:
            logger.error(f"Missing column in analytics model: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"The analytics model expected a column that wasn't provided: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Model execution failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Model execution error: {str(e)}"
            )
        
        # Process and return results as proper JSON
        if isinstance(results, pd.DataFrame):
            return results.applymap(convert_numpy_types).to_dict(orient='records')
        elif isinstance(results, list):
            return [df.applymap(convert_numpy_types).to_dict(orient='records') for df in results]
        else:
            return convert_numpy_types(results)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    return {
        "message": "Project Economics API",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "note": "All column mappings are properly handled (location→Country, baseYear→Base_Yr, CarbonTAX_value→CO2price)"
    }

@app.get("/health")
async def health_check():
    try:
        test_data = ProjectInput()
        load_multipliers()
        create_project_data(test_data)
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")
