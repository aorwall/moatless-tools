from pydantic import BaseModel

class LoopRequestDTO(BaseModel):
    agent_id: str
    model_id: str
    message: str
    # Attachments (file uploads) are provided as form-data and are not part of the JSON payload

class LoopResponseDTO(BaseModel):
    run_id: str 