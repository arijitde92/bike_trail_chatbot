# textbase/trail.py
from pydantic import BaseModel


class Trail(BaseModel):
    name: str
    id: int
    name: str
    url: str
    length: str
    description: str
    directions: str
    city: str
    region: str
    country: str
    difficulty: str
    features: str
    rating: int
