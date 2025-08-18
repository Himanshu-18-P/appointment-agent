from core.utils.handle_data import * 
from core.oai.llm import *

class ProcessApi:

    def __init__(self):
        self._handle_data = HandleData()
        self._process_text = ProcessInputText()

if __name__ == '__main__':
    print('done')