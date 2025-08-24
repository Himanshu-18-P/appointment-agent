from core.oai.tools import *  # Make sure `tools` uses the correct folder context
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.agents import initialize_agent, AgentType
import os


class ProcessInputText:
    def __init__(self):
        self.agents = {}  
        self.memories = {} 

    def get_or_create_agent(self, bot_name: str , system_prompt: str , api_key:str):
        """
        Get or initialize the agent and memory for a given bot_name (folder).
        """
        if bot_name in self.agents:
            return self.agents[bot_name]

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True ,
            output_key="output" , k = 10)

        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            api_key=api_key
        )
       

        agent = initialize_agent(
            tools=tools(bot_name), 
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            memory=memory,
            agent_kwargs={
                "system_message": SystemMessage(
                    content=(
                        system_prompt + f"\n\n[please use this as BotName: {bot_name}]"
                    )
                ),
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
            },
        )

        self.agents[bot_name] = agent
        self.memories[bot_name] = memory
        return agent

    def process(self, bot_name: str, user_input: str  , system_prompt : str , api_key : str) -> str:
        """
        Process user input using the agent specific to the given bot_name.
        """
        print('*'*10)
        print(self.memories[bot_name])
        print('*'*10)
        agent = self.get_or_create_agent(bot_name , system_prompt , api_key )
        return agent.invoke({"input": user_input})["output"]
    
    def process_stream(self, bot_name: str, user_input: str, system_prompt: str, api_key: str):
        """
        Stream the agent's reasoning steps and final output for the given bot_name.
        Yields chunks suitable for SSE/WebSocket or console output.
        """
        agent = self.get_or_create_agent(bot_name, system_prompt, api_key)
        
        for step in agent.stream({"input": user_input}):
            # `step` is a dict that can include "thought", "tool", "tool_input", etc.
            # Yield each step as a JSON-serializable dict or formatted text
            yield step


if __name__ == '__main__':
    print('done')