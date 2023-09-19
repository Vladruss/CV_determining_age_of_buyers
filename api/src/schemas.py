from pydantic import BaseModel, Field
from typing import List, Literal


class AgePrediction(BaseModel):
    id:str
    image_url: str


class AgePredictionRequest(BaseModel):
    images: List[AgePrediction] 


class AgePredicted(BaseModel):
    id:str
    age: int


class AgePredictionResult(BaseModel):
    result: List[AgePredicted] = Field(default_factory=list)