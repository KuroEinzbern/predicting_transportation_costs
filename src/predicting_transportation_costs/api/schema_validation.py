from pydantic import BaseModel
import datetime
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

class PredictionRequest(BaseModel):
    load_id : str = Field(pattern=r"^TR-\d+", max_length=20, description="ID of the transportation, expected format TR-XXXX")
    pickup : str = Field(max_length=100, description="Name of the city where the load was charged")
    delivery : str = Field(max_length=100, description="Name of the city where to deliver the payload")
    equipment : str = Field(max_length=100, description="brand of the payload")
    date : datetime.date = Field(description="Expected format yyyy-mm-dd")
    pickup_lat: float = Field(ge=-90, le=90, description= "latitud of the coordinates of the pick up point, should be between -90 and 90")
    pickup_lon : float = Field(ge=-180, le=180, description= "longitud of the coordinates of the pick up point, should be between -180 and 180")
    delivery_lat : float = Field(ge=-90, le=90, description= "latitud of the coordinates of the deliver point, should be between -90 and 90")
    delivery_lon : float = Field(ge=-180, le=180, description= "longitud of the coordinates of the delivery point, should be between -180 and 180")
    distance : float = Field(gt=0, description="total distance of the travel")
    weight : float = Field(gt=0, description= "weight of the package")
    market_index : float = Field(gt=0)
    quote_signal : float = Field(gt=0)


class PredictionResponse(BaseModel):
    load_id : str 
    prediction: float


