from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import os, json
import uvicorn
from typing import Optional
from langchain.schema import AIMessage
from core import *

app = FastAPI()
BASE_DIR = "bots_data"
processapi = ProcessApi()


class BotInitRequest(BaseModel):
    bot_name: str
    greeting: str = "👋 Hello! I'm your assistant."
    system_prompt: str = "You are a helpful assistant for scheduling doctor appointments."
    api_key : str = "sk- ........."


class UserMessage(BaseModel):
    message: str



@app.post("/bots/create")
def create_bot(bot_data: BotInitRequest):
    folder = processapi._handle_data.get_bot_folder(bot_data.bot_name)
    if os.path.exists(folder):
        raise HTTPException(status_code=400, detail="Bot already exists.")
    
    final_prompt = bot_data.system_prompt.strip() + "\n\n" + BASE_SYSTEM_PROMPT.strip()
    meta = {
        "greeting": bot_data.greeting,
        "system_prompt": final_prompt ,
        "api_key" : bot_data.api_key
    }

    processapi._handle_data.savejson(bot_data.bot_name , meta )

    return {"message": f"Bot '{bot_data.bot_name}' created."}


@app.post("/bots/{bot_name}/upload_schedule")
def upload_schedule(bot_name: str, file: UploadFile = File(...)):
    folder = processapi._handle_data.get_bot_folder(bot_name)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="Bot does not exist.")

    df = pd.read_csv(file.file)
    expected_cols = {"date", "time", "is_booked", "patient_name"}
    if not expected_cols.issubset(df.columns):
        raise HTTPException(status_code=400, detail=f"CSV must contain: {expected_cols}")

    df.to_csv(processapi._handle_data.get_schedule_path(bot_name), index=False)
    return {"message": f"Schedule updated for bot '{bot_name}'."}

@app.post("/bots/{bot_name}/upload_context_pdf")
def upload_context_pdf(bot_name: str, file: UploadFile = File(...)):
    folder = processapi._handle_data.get_bot_folder(bot_name)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="Bot does not exist.")

    # Save uploaded file
    pdf_path = os.path.join(folder, "context.pdf")
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    res = processapi.create_bot(bot_name , pdf_path , True)

    return {"message": f"Context PDF uploaded for bot '{bot_name}'."}



@app.get("/bots/{bot_name}/start")
def start_bot(bot_name: str):
    meta_path = processapi._handle_data.get_meta_path(bot_name)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Bot not found.")

    with open(meta_path, "r" , encoding="utf-8") as f:
        meta = json.load(f)

    greeting = meta.get("greeting", "👋 Hello! I'm your assistant.")
    system_prompt = meta.get("system_prompt", "You are a helpful assistant.")
    api_key = meta.get("api_key")

    # Initialize agent and inject greeting into memory
    agent = processapi._process_text.get_or_create_agent(bot_name, system_prompt, api_key)
    memory = processapi._process_text.memories[bot_name]
    memory.chat_memory.messages = [AIMessage(content=greeting)]

    return {"message": greeting}

@app.post("/bots/{bot_name}/chat")
def chat_with_bot(bot_name: str, user_message: UserMessage):
    folder_path = os.path.join(BASE_DIR, bot_name)
    config_path = os.path.join(folder_path, "meta.json")
    schedule_path = os.path.join(folder_path, "schedule.csv")

    if not os.path.exists(config_path) or not os.path.exists(schedule_path):
        raise HTTPException(status_code=404, detail="Bot configuration or schedule not found.")

    # Load prompt & initial message
    with open(config_path, "r" , encoding="utf-8") as f:
        config = json.load(f)

    response = processapi._process_text.process(bot_name,user_message.message , config.get('system_prompt') ,  config.get('api_key'))

    return {
        "bot_reply": response
    }


if __name__ == "__main__":
    uvicorn.run("main:app", port=8838, reload=True)
