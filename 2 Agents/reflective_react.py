import os
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------
# 1. HERRAMIENTAS Y MODELOS EN CASCADA
# ---------------------------------------------------------

@tool
def buscar_info_interna(query: str) -> str:
    """Busca datos internos de la empresa."""
    return f"Resultado de {query}: Operación exitosa."

tools = [buscar_info_interna]

# MODELO RÁPIDO: Se encarga del razonamiento repetitivo y uso de herramientas
fast_react_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).bind_tools(tools)

# MODELO ROBUSTO: Se encarga EXCLUSIVAMENTE de evaluar la calidad final
robust_evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ---------------------------------------------------------
# 2. ESQUEMA PYDANTIC PARA EVALUACIÓN ESTRICTA
# ---------------------------------------------------------
class EvaluationResult(BaseModel):
    is_valid: bool = Field(description="True si la respuesta resuelve la petición del usuario de forma precisa, False si faltan datos o hay alucinaciones.")
    feedback: str = Field(description="Instrucción corta de mejora si es False. Si es True, deja vacío.")

evaluator_chain = robust_evaluator_llm.with_structured_output(EvaluationResult)

# ---------------------------------------------------------
# 3. ESTADO DEL GRAFO
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    is_valid: bool

# ---------------------------------------------------------
# 4. DEFINICIÓN DE NODOS
# ---------------------------------------------------------
def react_agent_node(state: AgentState):
    """Nodo ReAct impulsado por el modelo rápido."""
    # Si hay feedback del evaluador en el estado (iteraciones > 0), el LLM lo verá en su contexto de mensajes
    response = fast_react_llm.invoke(state["messages"])
    
    # Inicializamos iterations en 1 en la primera pasada
    current_iterations = state.get("iterations", 0)
    if current_iterations == 0:
        current_iterations = 1
        
    return {"messages": [response], "iterations": current_iterations}

def evaluator_node(state: AgentState):
    """Nodo Evaluador impulsado por el modelo robusto + Pydantic."""
    # Extraemos la última respuesta generada por el agente ReAct
    draft_message = state["messages"][-1].content
    
    # Construimos un prompt ligero para no saturar al evaluador con tool_calls
    eval_prompt = f"Analiza esta respuesta final propuesta: {draft_message}. ¿Responde a la solicitud inicial correctamente y sin inventar datos?"
    
    # Forzamos la salida al esquema de Pydantic
    evaluation: EvaluationResult = evaluator_chain.invoke([HumanMessage(content=eval_prompt)])
    
    # Si no es válido, devolvemos el feedback como un mensaje humano para que el Agente ReAct lo corrija
    if not evaluation.is_valid:
        feedback_msg = HumanMessage(content=f"Crítica del evaluador: {evaluation.feedback}. Corrige tu respuesta y usa herramientas si es necesario.")
        return {
            "messages": [feedback_msg],
            "is_valid": False,
            "iterations": state["iterations"] + 1
        }
    
    return {"is_valid": True}

# ---------------------------------------------------------
# 5. LÓGICA DE ENRUTAMIENTO (ROUTERS)
# ---------------------------------------------------------
def route_from_react(state: AgentState) -> Literal["tools", "evaluator"]:
    """Decide si ir a las herramientas o enviar el borrador al evaluador."""
    last_message = state["messages"][-1]
    # Si el LLM solicitó usar una herramienta, vamos al ToolNode
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Si no pidió herramientas, significa que redactó una respuesta final -> Va a revisión
    return "evaluator"

def route_from_evaluator(state: AgentState) -> Literal["react", "__end__"]:
    """Decide si terminar o forzar una nueva iteración."""
    # Condición de salida exitosa o límite máximo de iteraciones (2 intentos) alcanzado
    if state.get("is_valid") or state["iterations"] >= 2:
        return END
    return "react"

# ---------------------------------------------------------
# 6. CONSTRUCCIÓN DEL GRAFO
# ---------------------------------------------------------
builder = StateGraph(AgentState)

builder.add_node("react", react_agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("evaluator", evaluator_node)

builder.add_edge(START, "react")

# En lugar de usar prebuilt.tools_condition, usamos nuestro propio router 
# para desviar el flujo al "evaluator" en vez de a END.
builder.add_conditional_edges("react", route_from_react)

builder.add_edge("tools", "react")

# Borde que evalúa si el agente logró superar la revisión
builder.add_conditional_edges("evaluator", route_from_evaluator, {"react": "react", END: END})

graph = builder.compile()