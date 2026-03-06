import math
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """
    Evaluate  a mathematical expression and return result.
    Use this for any arithmetic, percentages, square roots,
    trigonometry, or other maths calculations.
    Example: '2 + 2'
    """

    try:
        allowed_names = {
            k: v for k,v in math.__dict__.items()
            if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"