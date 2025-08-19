# appointment-agent

A no-code, API-first system to deploy intelligent doctor appointment assistants powered by LLMs.

Each bot supports:

* Personalized greetings
* Custom system prompts
* Its own appointment schedule
* Optional PDF knowledge base for contextual Q\&A


## 🚀 Features

- **Multi-Bot Support**  
  Easily create multiple assistants with unique schedules, prompts, and personalities.

- **LLM Integration**  
  Powered by GPT-4 with memory support via LangChain.

- **Appointment Handling**
  - Book appointments
  - Check availability
  - List free time slots

- **PDF Context Support** *(optional)*  
  Upload a PDF with doctor/clinic info and ask questions like:  
  _“Where is the clinic?”_ or _“Who is the doctor?”_

- **Natural Language Understanding**  
  Converts vague inputs like “tomorrow at 9” or “day after evening” using built-in time parsing.

- **Stateful Conversations**  
  Remembers patient names and preferences across the conversation.

- **Tool Calling with Safety**
  - Ensures required fields like `patient_name`, `date`, and `time` are confirmed before proceeding.
  - Prevents hallucination or over-assumption.

- **No-Code Friendly APIs**  
  Every function is exposed via clear, RESTful endpoints for easy integration into apps, websites, or no-code platforms.

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/appointment-agent.git

cd appointment-agent


python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Dependencies

pip install -r requirements.txt

# Set Environment Variables .env

OPENAI_API_KEY=sk-...

# Run the Server

uvicorn main:app --reload --port 8838

