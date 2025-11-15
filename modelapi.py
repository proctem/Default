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

def clean_dataframe_for_json(df):
    """Clean DataFrame to remove NaN, Inf, and other non-JSON-serializable values"""
    try:
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Replace NaN and Inf with None (which becomes null in JSON)
        df_clean = df_clean.replace([np.nan, np.inf, -np.inf], None)
        
        # Check for any remaining problematic values
        problematic_columns = []
        for col in df_clean.columns:
            # Check for any remaining non-serializable values
            for idx, value in enumerate(df_clean[col]):
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    problematic_columns.append((col, idx, value))
                    # Replace problematic value with None
                    df_clean.at[idx, col] = None
        
        if problematic_columns:
            logger.warning(f"Found problematic values in columns: {problematic_columns}")
            
        return df_clean
        
    except Exception as e:
        logger.error(f"Error cleaning dataframe: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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

        logger.info(f"Running analysis for: {request.location}, {request.product}, {request.plant_mode}, {request.fund_mode}")
        
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
        
        # Log the raw results for debugging
        logger.info(f"Raw results type: {type(results)}")
        if hasattr(results, 'shape'):
            logger.info(f"Raw results shape: {results.shape}")
        if hasattr(results, 'columns'):
            logger.info(f"Raw results columns: {results.columns.tolist()}")
        
        # Check for NaN values in the results - FIXED VERSION
        if hasattr(results, 'isna'):
            nan_count = results.isna().sum().sum()
            logger.info(f"NaN values in results: {nan_count}")
            
            # Get numeric columns only for Inf check
            numeric_results = results.select_dtypes(include=[np.number])
            inf_count = np.isinf(numeric_results).sum().sum()
            logger.info(f"Inf values in results: {inf_count}")
            
            if nan_count > 0 or inf_count > 0:
                # Log which columns have NaN values
                nan_columns = results.columns[results.isna().any()].tolist()
                logger.warning(f"Columns with NaN values: {nan_columns}")
                
                # Log which numeric columns have Inf values
                inf_columns = numeric_results.columns[np.isinf(numeric_results).any()].tolist()
                logger.warning(f"Columns with Inf values: {inf_columns}")
                
                # Log sample of problematic data for NaN
                for col in nan_columns:
                    nan_indices = results[col].isna()
                    if nan_indices.any():
                        sample_nan = results[col][nan_indices].head(3)
                        logger.warning(f"Sample NaN values in {col}: indices {nan_indices[nan_indices].index.tolist()[:3]}")
                        
                # Log sample of problematic data for Inf
                for col in inf_columns:
                    inf_indices = np.isinf(results[col])
                    if inf_indices.any():
                        sample_inf = results[col][inf_indices].head(3)
                        logger.warning(f"Sample Inf values in {col}: {sample_inf.tolist()}")
        
        # Clean the dataframe for JSON serialization
        results_clean = clean_dataframe_for_json(results)
        
        # Try to convert to dict and catch any serialization errors
        try:
            results_dict = results_clean.to_dict(orient='records')
            logger.info(f"Successfully converted results to dict with {len(results_dict)} records")
        except Exception as e:
            logger.error(f"Error converting results to dict: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing results: {str(e)}")
        
        # Test JSON serialization
        try:
            test_json = json.dumps(results_dict)
            logger.info("Results successfully serialized to JSON")
        except Exception as e:
            logger.error(f"JSON serialization error: {str(e)}")
            # Try to identify the problematic value
            for i, record in enumerate(results_dict):
                try:
                    json.dumps(record)
                except Exception as record_error:
                    logger.error(f"Problematic record {i}: {record_error}")
                    # Log which key-value pair is problematic
                    for key, value in record.items():
                        try:
                            json.dumps(value)
                        except Exception as value_error:
                            logger.error(f"Problematic key-value pair: {key} = {value} (error: {value_error})")
                    break
            raise HTTPException(status_code=500, detail=f"JSON serialization failed: {str(e)}")
        
        return results_dict

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "data_loaded": PROJECT_DATA is not None and MULTIPLIER_DATA is not None,
        "project_data_rows": len(PROJECT_DATA) if PROJECT_DATA is not None else 0,
        "multiplier_data_rows": len(MULTIPLIER_DATA) if MULTIPLIER_DATA is not None else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
