"""
simple_reflective_react.py
El nodo ReAct piensa y decide si usar herramientas o dar una respuesta borrador, mientras que
el nodo Evaluador analiza la respuesta generada por el agente ReAct
"""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage


@tool
def buscar_info_interna(query: str) -> str:
    """Busca datos internos de la empresa."""
    return f"Resultado de {query}: Operación exitosa."


tools = [buscar_info_interna]


class MessagesState(TypedDict):
    is_valid: bool
    messages: str


def react_agent_node(state):
    # Lógica ReAct normal...
    pass


def evaluator_node(state):
    # Analiza si la respuesta responde correctamente a la solicitud original
    # y si usó bien los datos de las herramientas
    pass


# 3. Router para decidir si terminar o re-intentar
def route_after_evaluation(state):
    if state.get("is_valid"):
        return END
    return "react_agent"  # Si no es válida, regresa al agente con feedback


builder = StateGraph(State)

builder.add_node("react_agent", react_agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("evaluator", evaluator_node)

builder.add_edge(START, "react_agent")

# Borde condicional ReAct: va a herramientas o al evaluador
builder.add_conditional_edges(
    "react_agent",
    tools_condition,
    {
        "tools": "tools",       # Si necesita ejecutar herramientas
        END: "evaluator"        # Si terminó sus llamadas a herramientas, pasa a revisión
    }
)

builder.add_edge("tools", "react_agent")

# Borde condicional del Evaluador
builder.add_conditional_edges("evaluator", route_after_evaluation)

graph = builder.compile()
