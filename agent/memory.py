
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from config.setting import MAX_HISTORY_MESSAGES

class AgentMemory:
    """
    Wraps LangChain's conversationBufferWindowMemory.
    
    ConversationBufferWindowMemory keeps the last K exchanges 
    in memory - exactly what our MAX_HISTORY_MESSAGES setting controls.
    
    LangChain memory automatically handles: 
      - Storing user messages
      - Storing AI responses
      - Formatting history for the LLM
      - Trimming old messages (via the k parameter) 
    """

    def __init__(self,system_prompt: str= ""):
        # k = number of conversation Exchanges to keep (1 exchange = 1 user + 1 AI)
        # so k=10 keeps 20 messages in total
        self.memory = ConversationBufferWindowMemory(
            k = MAX_HISTORY_MESSAGES //2,
            memory_key="chat_history",     # key used in the prompt templates
            return_messages=True,          # return message objects, not strings
            output_key="output"
        )
        self.system_prompt =system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are a helpful AI assistant with access to tools."
            "Use tools when you need real time information or to perform actions. "
            "Think step by step. Be concise, accurate and helpful. "

        )
    def get_memory(self) -> ConversationBufferWindowMemory:
        """Return the raw langchain memory object (Pass the AgentExecutor). """
        return self.memory

    def get_history(self) -> list:
        """Return message history as a list"""
        return self.memory.chat_memory.messages

    def clear(self):
        """Reset conversation history."""
        self.memory.clear()

    def __len__(self):
        return len(self.memory.chat_memory.messages)
