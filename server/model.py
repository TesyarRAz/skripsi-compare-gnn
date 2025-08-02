from pydantic import BaseModel

class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class CoordLocation(BaseModel):
    lat: float
    lon: float

class PredictCoordTour(BaseModel):
    model: str
    coords: list[CoordLocation]

    class Config:
        json_schema_extra = {
            "example": {
                "model": "gat/10_50",
                "coords": [
                    {"lat": -6.2, "lon": 106.8},
                    {"lat": -7.0, "lon": 107.6},
                    {"lat": -6.9, "lon": 106.7},
                    {"lat": -6.5, "lon": 106.9},
                    {"lat": -7.1, "lon": 107.0},
                    {"lat": -6.8, "lon": 106.5},
                    {"lat": -7.2, "lon": 107.1},
                    {"lat": -6.7, "lon": 106.6},
                    {"lat": -6.4, "lon": 106.8},
                    {"lat": -7.3, "lon": 107.2}
                ]
            }
        }