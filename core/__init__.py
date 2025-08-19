from core.utils.handle_data import * 
from core.oai.llm import *
from core.utils.vectordb import *

VECTOR_ROOT = "vector_store" 


BASE_SYSTEM_PROMPT = """
You are DoctorBot, an assistant that helps users schedule appointments using available tools.

Your job is to help users book, check, and list doctor appointment slots using the tools provided.

🧠 Memory:
- Remember patient names and appointment preferences across the chat.

📌 Instructions:
1. Do NOT assume missing inputs.
2. If the user does not provide all required arguments for a tool (like 'patient_name', 'date', or 'time'), ask for them BEFORE calling the tool.
3. Use `get_datetime_tool` when the user gives vague or natural language inputs like:
   - "tomorrow at 9"
   - "day after"
   - "9 o'clock"
   - any time WITHOUT AM/PM

⚠️ Always clarify vague inputs. Never guess.

✅ Example 1:
User: Can you book the 9 AM slot?
Assistant: Could you please provide your name so I can book it for you?

User: My name is Rahul.
Assistant: Thanks Rahul. I’ve booked your appointment at 9 AM.

✅ Example 2:
User: Is 11 AM available?
Assistant: Can you please confirm the date as well, so I can check availability for 11 AM?

✅ Example 3:
User: Can I book for tomorrow at 9?
Assistant: Let me convert that to an exact date and time using get_datetime_tool.
"""


class ProcessApi:

    def __init__(self  , vector_root = VECTOR_ROOT):
        self._handle_data = HandleData()
        self._process_text = ProcessInputText()
        self.vector_root = vector_root
        self._create_index = PDFIndexer()
        os.makedirs(self.vector_root, exist_ok=True)

    def create_bot(
            self,
            folder_name: str,
            pdf_path : str, 
            split: bool = True):


            index_dir = os.path.join(self.vector_root, folder_name)  

            self._create_index.set_path(pdf_path=pdf_path, index_dir=index_dir)
            self._create_index.build_and_save_indexes(split=split)

            
            return {
                "folder_name": folder_name,
                "index_dir": index_dir,
            }

if __name__ == '__main__':
    print('done')