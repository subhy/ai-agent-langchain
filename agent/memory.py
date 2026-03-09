from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory  # or use your own persistent one

class AgentMemory:
    def __init__(self, window_size: int = 10):
        # In-memory history (replace with Redis, DynamoDB, Filesystem, etc. for persistence)
        self.chat_history = ChatMessageHistory()

        # Optional: enforce a window by manually trimming (simple implementation)
        self.window_size = window_size  # number of messages to keep (or pairs)

        self.system_prompt = (
            "You are a helpful AI assistant. Answer concisely and accurately."
        )

    def get_memory(self) -> BaseChatMessageHistory:
        """Return the history object to pass to RunnableWithMessageHistory"""
        return self.chat_history

    def clear(self):
        """Clear conversation history"""
        self.chat_history.clear()

    def trim_if_needed(self):
        """Simple window: keep only the last N messages (optional)"""
        messages = self.chat_history.messages
        if len(messages) > self.window_size:
            # Keep system prompt if present + last window_size messages
            self.chat_history.messages = messages[-self.window_size:]

    def add_user_message(self, content: str):
        self.chat_history.add_message(HumanMessage(content=content))
        self.trim_if_needed()

    def add_ai_message(self, content: str):
        self.chat_history.add_message(AIMessage(content=content))
        self.trim_if_needed()

    def __len__(self):
        return len(self.chat_history.messages)