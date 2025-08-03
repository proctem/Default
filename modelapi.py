from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional, List
import uvicorn
from originalmodel import  Analytics_Model2

app = FastAPI(
    title="Integrated Project Economics Model API",
    description="API for analyzing chemical plant economics with micro and macroeconomic impacts",
    version="1.0.0"
)

# Load data files at startup
@app.on_event("startup")
async def startup_event():
    global project_datas, multipliers
    try:
        project_datas = pd.read_csv("./project_data.csv")
        multipliers = pd.read_csv("./sectorwise_multipliers.csv")
    except FileNotFoundError as e:
        raise Exception(f"Required data files not found: {str(e)}")

# Input validation models
class AnalysisRequest(BaseModel):
    location: str
    product: str
    plant_mode: str = "Green"  # Green or Brown
    plant_size: Optional[str] = None  # Large or Small
    plant_effy: Optional[str] = None  # High or Low
    fund_mode: str = "Equity"  # Debt, Equity, or Mixed
    opex_mode: str = "Inflated"  # Inflated or Uninflated
    carbon_value: str = "Yes"  # Yes or No

@app.get("/")
async def root():
    return {
        "message": "Integrated Project Economics Model API",
        "endpoints": {
            "/analyze": "POST - Run economic analysis with parameters",
            "/locations": "GET - List available countries",
            "/products": "GET - List available products"
        }
    }

@app.get("/locations")
async def get_locations():
    """Get list of available countries/locations"""
    locations = project_datas['Country'].unique().tolist()
    return {"locations": locations}

@app.get("/products")
async def get_products():
    """Get list of available products"""
    products = project_datas['Main_Prod'].unique().tolist()
    return {"products": products}

@app.post("/analyze", response_model=List[dict])
async def run_analysis(request: AnalysisRequest):
    """
    Run the integrated economic analysis with the provided parameters
    
    Parameters:
    - location: Country where plant is located
    - product: Chemical product being produced
    - plant_mode: "Green" (new plant) or "Brown" (existing plant)
    - plant_size: Optional filter for plant size ("Large" or "Small")
    - plant_effy: Optional filter for plant efficiency ("High" or "Low")
    - fund_mode: Project financing mode ("Debt", "Equity", or "Mixed")
    - opex_mode: Operating expense mode ("Inflated" or "Uninflated")
    - carbon_value: Whether to include carbon costs ("Yes" or "No")
    """
    # Validate inputs
    if request.location not in project_datas['Country'].unique():
        raise HTTPException(status_code=400, detail="Invalid location")
    
    if request.product not in project_datas['Main_Prod'].unique():
        raise HTTPException(status_code=400, detail="Invalid product")
    
    if request.plant_mode not in ["Green", "Brown"]:
        raise HTTPException(status_code=400, detail="plant_mode must be 'Green' or 'Brown'")
    
    if request.fund_mode not in ["Debt", "Equity", "Mixed"]:
        raise HTTPException(status_code=400, detail="fund_mode must be 'Debt', 'Equity', or 'Mixed'")
    
    if request.opex_mode not in ["Inflated", "Uninflated"]:
        raise HTTPException(status_code=400, detail="opex_mode must be 'Inflated' or 'Uninflated'")
    
    if request.carbon_value not in ["Yes", "No"]:
        raise HTTPException(status_code=400, detail="carbon_value must be 'Yes' or 'No'")
    
    # Filter data based on request
    filter_conditions = [
        (project_datas['Country'] == request.location),
        (project_datas['Main_Prod'] == request.product)
    ]
    
    if request.plant_size:
        if request.plant_size not in ["Large", "Small"]:
            raise HTTPException(status_code=400, detail="plant_size must be 'Large' or 'Small'")
        filter_conditions.append(project_datas['Plant_Size'] == request.plant_size)
    
    if request.plant_effy:
        if request.plant_effy not in ["High", "Low"]:
            raise HTTPException(status_code=400, detail="plant_effy must be 'High' or 'Low'")
        filter_conditions.append(project_datas['Plant_Effy'] == request.plant_effy)
    
    filtered_data = project_datas[np.logical_and.reduce(filter_conditions)]
    
    if len(filtered_data) == 0:
        raise HTTPException(status_code=404, detail="No project data matches the specified criteria")
    
    # Run analysis
    try:
        results = Analytics_Model2(
            multiplier=multipliers,
            project_data=filtered_data,
            location=request.location,
            product=request.product,
            plant_mode=request.plant_mode,
            fund_mode=request.fund_mode,
            opex_mode=request.opex_mode,
            plant_size=request.plant_size,
            plant_effy=request.plant_effy,
            carbon_value=request.carbon_value
        )
        
        # Convert results to list of dicts for JSON response
        return results.to_dict(orient='records')
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running analysis: {str(e)}")

# Include all your model functions here (ChemProcess_Model, MicroEconomic_Model, MacroEconomic_Model, Analytics_Model2)
# ... [paste all the model functions from your original code here] ...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
