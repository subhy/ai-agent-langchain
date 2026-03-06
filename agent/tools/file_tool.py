import os
from os.path import exists

from langchain_core.tools import tool

@tool
def read_file(filepath: str) -> str:
    """
    Read and return the content of a text file from the filesystem.
    Use this when the user wants to view or analyze a file.
    Input should be the path to the file.
    """
    try:
        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath):
            return f"Error: File not found at '{filepath}'"

        with open(filepath,"r",encoding="utf-8") as f:
            content = f.read()

        return content if content.strip() else "File exists but is empty"

    except PermissionError:
        return f"Error: Permission denied to read '{filepath}'"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file on the filesystem.
    Create the file and any parent directories if they don't exist.
    Use this when the user wants to save, create, or update a file.
    Inputs: filepath (where to save), content (what to write).
    """

    try:
        filepath = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to '{filepath}'"
    except PermissionError:
        return f"Error: Permission denied to write '{filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"



