from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import math
import json
# Connect Ollama model
ollama_llm = ChatOllama(model="llama3.2")




#Creating first agent (Arithmatic Calculator)
#tool
def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return (a * b)

def divide(a: float, b: float) -> float:
    if b == 0:
        return "Error: Division by zero"
    return a / b
@tool
def add_tool(a: float, b: float) -> float:
    """Add two numbers"""
    return add(a, b)

@tool
def multiply_tool(a: float, b: float) -> float:
    """Multiply two numbers"""
    return multiply(a, b)
@tool
def subtract_tool(a: float, b: float) -> float:
    """Add two numbers"""
    return subtract(a, b)

@tool
def divide_tool(a: float, b: float) -> float:
    """Multiply two numbers"""
    return divide(a, b)
@tool
def exponentiate_tool(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent
simple_maths_agent = create_react_agent(
    model=ollama_llm,
    tools=[add_tool, subtract_tool, multiply_tool, divide_tool,exponentiate_tool],
    prompt=(
        "You are a strict math calculator agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Solve ONLY basic arithmetic (addition, subtraction, multiplication, division) using BODMAS.\n"
        "- NEVER explain your steps, NEVER show intermediate reasoning.\n"
        "- ALWAYS call the correct tool for every operation.\n"
        "- DO NOT write text, notes, or explanations.\n"
        "- Your final answer must ONLY be the computed numeric result.\n"
        "- If parentheses are present, evaluate them first strictly using tools.\n"
        "- If you cannot solve, return 'ERROR'."
    ),
    name="simple_maths_agent",
)
# Creating Agent 2(Counting problems agent)

#Tools
@tool
def factorial_tool(n: int) -> int:
    """Return n! (n factorial)."""
    return math.factorial(n)

@tool
def permutation_tool(n: int, r: int) -> int:
    """Return number of permutations: P(n, r)."""
    return math.factorial(n) // math.factorial(n - r)

@tool
def combination_tool(n: int, r: int) -> int:
    """Return number of combinations: C(n, r)."""
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

counting_agent = create_react_agent(
    model=ollama_llm,
    tools=[factorial_tool, permutation_tool, combination_tool],
    prompt=(
        "You are a probability agent.\n\n"
        "INSTRUCTIONS:\n"
        "-Break down problems into simple problems first "
        "- Solve problems involving counting, permutations, combinations, and factorials.\n"
        "- Use the provided tools (factorial, permutation, combination) when required.\n"
        "- Follow correct probability and combinatorial reasoning.\n"
        "- After completing calculations, respond directly with the final result.\n"
        "- Do NOT explain steps unless explicitly asked.\n"
    ),
    name="counting_agent",
)


# Creating a Supervisor Agent

from langgraph_supervisor import create_supervisor

# ---------------- SUPERVISOR ---------------- #
supervisor = create_supervisor(
    model=ollama_llm,
    agents=[simple_maths_agent, counting_agent],
    prompt=(
   "You are a supervisor managing two agents:\n"
    "- a math agent (simple_maths_agent). Assign only arithmetic tasks to this agent.\n"
    "- a combinatorics agent (counting_agent). Assign only counting, factorial, permutation, or combination tasks to this agent.\n"
    "Firstly, break down complex problems into simpler sub-problems before assigning.\n"
    "Assign work to one agent at a time. Do not call agents in parallel.\n"
    "Do not perform any calculations yourself. Only coordinate the agents.\n"
    "Always enforce the use of the agent tools.\n"
    "Always respond ONLY with the numeric result of the calculation. Do NOT include any text, explanations, or messages.\n"
    "Format the answer exactly as: ANSWER: <number>\n"
    "If a query cannot be solved, return: ANSWER: ERROR"
    ),
    add_handoff_back_messages=True,
    output_mode="last_message",
).compile()

while True:
    user_query = input("Ask any Mathematical Problem (type 'exit' to quit): ")
    if user_query.lower() in ["exit", "quit"]:
        print("Exiting Calculator Agent.")
        break

    # result = supervisor.invoke({
    #     "messages": [{"role": "user", "content": user_query}]
    # })
    for step in supervisor.stream({"messages": [user_query]}, stream_mode="values"):
      step["messages"][-1].pretty_print()

   


