from fastapi import FastAPI
from pydantic import BaseModel
from Pipeline_Data.test_rag_copy import create_rag, get_answer

rag = create_rag()
app = FastAPI()

class UserInput(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: UserInput):
    answer = get_answer(data.text, rag)
    return {"text": answer}