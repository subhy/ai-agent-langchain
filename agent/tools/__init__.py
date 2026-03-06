from agent.tools.calculator import calculate
from agent.tools.search import web_search
from agent.tools.file_tool import read_file,write_file

# Single list - pass this directly to Langchain agent

ALL_TOOLS = [
    calculate,
    web_search,
    read_file,
    write_file,
]
