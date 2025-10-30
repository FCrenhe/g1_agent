from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def speak(text: str) -> str:
    """Speak the given text."""
    return "我已经朗读了" + text + "。"

@tool
def weather() -> str:
    """Returns the current weather conditions."""
    return "It's nice and sunny."


#test_tools

#print(multiply.invoke({"a": 2, "b": 3}))
#print(speak.invoke({"text": "天下无敌"}))