from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel


from config.setting import (
    LLM_PROVIDER,
    OPENAI_API_KEY,OPENAI_MODEL,
    ANTHROPIC_API_KEY,ANTHROPIC_MODEL,
    TEMPERATURE,MAX_ITERATIONS
)
from agent.memory import AgentMemory
from agent.tools import ALL_TOOLS

class Agent:
    """
    AI Agent build with LangChain.
    """

    def __init__(self):
        """
        Assemble the LangChain Agent pipeline.
        The pipeline is: LLM + Tools + Prompt + Memory + AgentExecutor
        AgentExecutor is the equivalent of entire _run_loop() method.
        """

        # 1. Get the LLM
        self.provider = LLM_PROVIDER
        self.memory = AgentMemory()
        llm =self._get_llm()

        # 2. Build the prompt template.
        # MessagesPlaceholder is a slot where LangChain injects
        # chat history and agent scratchpad automatically

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.memory.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # memory goes here
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad") #tool calls goes here.
        ])

        # 3. Create the ReAct agent
        # This is the reasoning engine - it's implement the
        # Thought -> Action -> observation loop.

        react_agent = create_react_agent(
            llm = llm,
            tools = ALL_TOOLS,
            prompt = prompt
        )

        # 4. Wrap in AgentExecutor
        #  This is the loop runner  - equivalent to while loop.
        self.executor = AgentExecutor(
            agent = react_agent,
            tools = ALL_TOOLS,
            memory = self.memory.get_memory(),
            max_iterations = MAX_ITERATIONS,    # MAX_ITERATION guard
            verbose = True,                     # prints Thought/Action/Observation
            handle_parsing_errors =True,        # gracefully handle LLM format errors
            return_intermediate_steps = True    # capture tool usage for logging
        )

    def _get_llm(self) -> BaseChatModel:
        """
        Return the correct LangChain LLM object based on provider.

        Both ChatOpenAI ChatAnthropic implement BaseChatModel.
        so the rest of the code never needs to know which one is active.
        This is the strategy pattern - same interface, swappable implementation.
        """
        if self.provider == "openai":
            return ChatOpenAI(
                api_key = OPENAI_API_KEY,
                model = OPENAI_MODEL,
                temperature= TEMPERATURE
            )
        elif self.provider == "anthropic":
            return ChatAnthropic(
                api_key =ANTHROPIC_API_KEY,
                model= ANTHROPIC_MODEL,
                temperature = TEMPERATURE
            )
        else:
            raise ValueError(
                f"Unknown provider: '{self.provider}'."
                f"Use 'openai' or 'anthropic'."
            )

        #------------ Public Interface -----------------------

    def chat(self,user_input: str) -> str:
        """
        Send a message and get a response.

        Internally this calls AgentExecutor.invoke() which:
        1. Formats the prompt with chat history.
        2. Calls the LLM.
        3. Parse the output. (tool call or final answer?)
        4. If tool: runs it, adds result, loop back to step 2
        5. If final answer: returns it
        6. Stops if MAX_ITERATIONS reached

        All of that happen in ONE line below.
        """
        try:
            result = self.executor.invoke({"input": user_input})
            return result["output"]
        except Exception as e:
            return f"Agent error: {str(e)}"

    def switch_provider(self,provider: str):
        """
        Switch LLM at runtime.
        Rebuilds the agent with the new LLM - memory is preserved.
        """
        self.provider =provider
        self._build_agent()  # rebuild with new LLM, same memory.
        print(f"[Agent] Switched to: {provider}")

    def reset(self):
        """
        Clear conversation history.
        """
        self.memory.clear()
        print("[Agent] Conversation reset.")

    def get_message_count(self) -> int:
        return len(self.memory)


