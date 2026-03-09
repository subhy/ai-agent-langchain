from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.history import RunnableWithMessageHistory

from config.setting import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    TEMPERATURE,
    MAX_ITERATIONS
)

from agent.memory import AgentMemory
from agent.tools import ALL_TOOLS


class Agent:
    """
    Modern LangChain agent using tool-calling + RunnableWithMessageHistory.
    Compatible with LangChain 0.3+ / 1.0+ (no more memory= deprecation issues).
    """

    def __init__(self):
        """Initialize LLM, memory, prompt, agent and executor."""
        # Provider & memory first
        self.provider = LLM_PROVIDER.lower().strip()
        self.memory = AgentMemory()           # contains .chat_history (BaseChatMessageHistory)

        # LLM
        self.llm = self._get_llm()

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.memory.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Tool-calling agent
        agent_runnable = create_tool_calling_agent(
            llm=self.llm,
            tools=ALL_TOOLS,
            prompt=self.prompt,
        )

        # Executor without memory= (legacy field)
        executor = AgentExecutor(
            agent=agent_runnable,
            tools=ALL_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=MAX_ITERATIONS,
            return_intermediate_steps=True,
        )

        # ─── Modern memory injection ───
        self.executor_with_history = RunnableWithMessageHistory(
            runnable=executor,
            get_session_history=lambda session_id: self.memory.get_memory(),
            input_messages_key="input",
            history_messages_key="chat_history",
            # Optional: if you later want per-user conversations
            # history_factory_config={"session_id": "configurable"}
        )

    def _get_llm(self) -> BaseChatModel:
        """Factory for current LLM."""
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            return ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=OPENAI_MODEL,
                temperature=TEMPERATURE,
            )
        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return ChatAnthropic(
                api_key=ANTHROPIC_API_KEY,
                model=ANTHROPIC_MODEL,
                temperature=TEMPERATURE,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _rebuild_agent(self):
        """Rebuild when provider changes (memory preserved)."""
        self.llm = self._get_llm()

        agent_runnable = create_tool_calling_agent(
            llm=self.llm,
            tools=ALL_TOOLS,
            prompt=self.prompt,
        )

        executor = AgentExecutor(
            agent=agent_runnable,
            tools=ALL_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=MAX_ITERATIONS,
            return_intermediate_steps=True,
        )

        self.executor_with_history = RunnableWithMessageHistory(
            runnable=executor,
            get_session_history=lambda session_id: self.memory.get_memory(),
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    # ────────────────────────────────────────────────
    # Public methods
    # ────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """Run one turn with history automatically handled."""
        try:
            config = {"configurable": {"session_id": "console_user_1"}}
            # Use the wrapped runnable (no config needed for single-user)
            result = self.executor_with_history.invoke(
                {"input": user_input.strip()},
                config = config
            )
            return result["output"].strip()
        except Exception as e:
            return f"Agent error: {str(e)}"

    def switch_provider(self, provider: str):
        """Switch LLM and rebuild (memory kept)."""
        provider = provider.lower().strip()
        if provider not in ("openai", "anthropic"):
            return f"[Agent] Unsupported provider: {provider}"

        if provider == self.provider:
            return f"[Agent] Already using {provider}"

        self.provider = provider
        self._rebuild_agent()
        return f"[Agent] Switched to {provider}"

    def reset(self):
        """Clear history."""
        self.memory.clear()
        return "[Agent] Conversation history cleared."

    def get_message_count(self) -> int:
        return len(self.memory)


if __name__ == "__main__":
    agent = Agent()
    print(f"Agent ready – provider: {agent.provider}")
    print(f"Tools: {len(ALL_TOOLS)}")