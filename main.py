from fastapi import FastAPI, UploadFile, File, HTTPException , Form
from pydantic import BaseModel
import pandas as pd
import os, json
import uvicorn
from typing import Optional
from langchain.schema import AIMessage
from core import *
import uuid
from fastapi.middleware.cors import CORSMiddleware
import re
from fastapi.responses import StreamingResponse
import json


app = FastAPI()
BASE_DIR = "bots_data"
processapi = ProcessApi()


# Allow all origins (NOT recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # Allow all headers (Authorization, Content-Type, etc.)
)

 
class BotInitRequest(BaseModel):
    bot_name: str
    greeting: str = "👋 Hello! I'm your assistant."
    api_key: Optional[str] = None

class UserMessage(BaseModel):
    message: str
    bot_name: str


def slugify(text: str) -> str:
    # Convert spaces and special chars into safe dashes/underscores
    return re.sub(r'[^a-zA-Z0-9_-]', '-', text.strip().lower())


@app.get('/')
def index():
    return {"message" : "Hare Krishna"}


@app.post("/bots/create")
def create_bot(bot_data: BotInitRequest):
    try:
        # Create a unique ID
        safe_name = slugify(bot_data.bot_name)
        unique_id = str(uuid.uuid4())[:8]   # short UUID for uniqueness
        bot_id = f"{safe_name}-{unique_id}"

        # Folder for this bot
        folder = processapi._handle_data.get_bot_folder(bot_id)

        if os.path.exists(folder):
            raise HTTPException(status_code=400, detail="Bot already exists.")

        final_prompt = BASE_SYSTEM_PROMPT.strip()
        meta = {
            "greeting": bot_data.greeting,
            "system_prompt": final_prompt,
            "api_key": bot_data.api_key,
            "bot_name": bot_data.bot_name,
            "bot_id": bot_id
        }

        processapi._handle_data.savejson(bot_id, meta)

        return {"message": bot_data.bot_name, "bot_id": bot_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.post("/bots/upload_schedule")
def upload_schedule(
    bot_name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        folder = processapi._handle_data.get_bot_folder(bot_name)
        if not os.path.exists(folder):
            raise HTTPException(status_code=404, detail="Bot does not exist.")

        df = pd.read_csv(file.file)
        expected_cols = {"date", "time", "is_booked", "patient_name"}
        if not expected_cols.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"CSV must contain: {expected_cols}")

        df.to_csv(processapi._handle_data.get_schedule_path(bot_name), index=False)
        return {"message": f"Schedule updated for bot '{bot_name}'."}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



@app.post("/bots/upload_context_pdf")
def upload_context_pdf(
    bot_name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        folder = processapi._handle_data.get_bot_folder(bot_name)
        if not os.path.exists(folder):
            raise HTTPException(status_code=404, detail="Bot does not exist.")

        pdf_path = os.path.join(folder, "context.pdf")
        with open(pdf_path, "wb") as f:
            f.write(file.file.read())

        res = processapi.create_bot(bot_name, pdf_path, True)

        return {"message": f"Context PDF uploaded for bot '{bot_name}'."}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



@app.get("/bots/{bot_name}/start")
def start_bot(bot_name: str):
    try:
        meta_path = processapi._handle_data.get_meta_path(bot_name)
        if not os.path.exists(meta_path):
            raise HTTPException(status_code=404, detail="Bot not found.")

        with open(meta_path, "r" , encoding="utf-8") as f:
            meta = json.load(f)

        greeting = meta.get("greeting", "👋 Hello! I'm your assistant.")
        system_prompt = BASE_SYSTEM_PROMPT
        api_key = meta.get("api_key")  or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="No API key provided or found in environment.")

        # Initialize agent and inject greeting into memory
        agent = processapi._process_text.get_or_create_agent(bot_name, system_prompt, api_key)
        memory = processapi._process_text.memories[bot_name]
        memory.chat_memory.messages = [AIMessage(content=greeting)]

        return {"message": greeting}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/bots/chat")
def chat_with_bot(user_message: UserMessage):
    try:
        folder_path = os.path.join(BASE_DIR, user_message.bot_name)
        config_path = os.path.join(folder_path, "meta.json")
        schedule_path = os.path.join(folder_path, "schedule.csv")

        if not os.path.exists(config_path) or not os.path.exists(schedule_path):
            raise HTTPException(status_code=404, detail="Bot configuration or schedule not found.")

        # Load prompt & initial message
        with open(config_path, "r" , encoding="utf-8") as f:
            config = json.load(f)

        response = processapi._process_text.process(user_message.bot_name,user_message.message , config.get('system_prompt') ,  config.get('api_key'))

        return {
            "bot_reply": response
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    


@app.post("/bots/stream")
def chat_with_bot_stream(user_message: UserMessage):
    try:
        folder_path = os.path.join(BASE_DIR, user_message.bot_name)
        config_path = os.path.join(folder_path, "meta.json")
        schedule_path = os.path.join(folder_path, "schedule.csv")

        if not os.path.exists(config_path) or not os.path.exists(schedule_path):
            raise HTTPException(status_code=404, detail="Bot configuration or schedule not found.")

        # Load prompt & initial message
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        def event_stream():
            try:
                for step in processapi._process_text.process_stream(
                    user_message.bot_name,
                    user_message.message,
                    config.get("system_prompt"),
                    config.get("api_key")
                ):
                    # Final Output
                    if "output" in step:
                        yield f"data: {json.dumps({'type': 'final', 'output': step['output']})}\n\n"

                    # Intermediate Reasoning + Tool Use
                    elif "steps" in step:
                        # Handle agent step — extract useful info
                        try:
                            action = step["steps"][0].action  # this is AgentActionMessageLog
                            observation = step["steps"][0].observation

                            # Create cleaned version
                            cleaned = {
                                "type": "tool_use",
                                "log": action.log.strip(),
                                "tool": action.tool,
                                "tool_input": action.tool_input,
                                "observation": observation
                            }
                            yield f"data: {json.dumps(cleaned)}\n\n"

                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"



        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



if __name__ == "__main__":
    uvicorn.run("main:app", port=8838,host='0.0.0.0' , reload=True)
