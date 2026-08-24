# basic_agent.py: Basic agent with LangGraph and without LLM
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()


class State(TypedDict):
    my_var: str
    customer_name: str


def func_1(state: State) -> State:
    state["my_var"] = "Hello from node 1"
    return state


def func_2(state: State) -> State:
    state["my_var"] = "Hello from node 2"
    return state


def func_3(state: State) -> State:
    state["my_var"] = "Hello from node 3"
    state["customer_name"] = "Name from node 3"
    return state


def decision_node(state: State) -> Literal['node_2', 'node_3']:
    if len(state["my_var"]) > 1:
        return 'node_2'
    else:
        return 'node_3'


builder = StateGraph(State)

builder.add_node('node_1', func_1)
builder.add_node('node_2', func_2)
builder.add_node('node_3', func_3)

builder.add_edge(START, 'node_1')
builder.add_conditional_edges('node_1', decision_node)
builder.add_edge('node_2', END)
builder.add_edge('node_3', END)

graph = builder.compile()
# graph.get_graph().draw_mermaid_png(output_file_path="./img/basic_agent.png")
