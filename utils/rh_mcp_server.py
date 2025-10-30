from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print("aaaaaa")
    return a + b

@mcp.tool()
def run() -> str:
    """run run run"""
    print("run run run")
    return "start run"

if __name__ == "__main__":
  #  print("run************")
    mcp.run(transport="stdio")