from typing import Dict, List
from pydantic import BaseModel, field_validator
import json

class Instance(BaseModel):
    courseCapabilities: Dict[str, int]
    agentCapabilities: Dict[str, int]
    bids: Dict[str, Dict[str, int]]
    courseOrderPerStudent: Dict[str, List[str]]
    tieBrakingLottery: Dict[str, float]
    
    @field_validator("courseCapabilities", "agentCapabilities", mode="before")
    @classmethod
    def convert_str_to_dict_int(cls, value: str) -> Dict[str, int]:
        if isinstance(value, str):
            return json.loads(value)

    @field_validator( "tieBrakingLottery", mode="before")
    @classmethod
    def convert_str_to_dict_float(cls, value: str) -> Dict[str, float]:
        if isinstance(value, str):
            return json.loads(value)
    
    @field_validator("bids", mode="before")
    @classmethod
    def convert_str_to_dict_dict(cls, value: str) -> Dict[str, Dict[str, int]]:
        if isinstance(value, str):
            return json.loads(value)
    
    @field_validator("courseOrderPerStudent", mode="before")
    @classmethod
    def convert_str_to_dict_list(cls, value: str) -> Dict[str, List[str]]:
        if isinstance(value, str):
            return json.loads(value)
