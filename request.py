from pydantic import BaseModel

class Recommend(BaseModel):
    user_id : str
    card_type : str 
    want_to_find : str
    