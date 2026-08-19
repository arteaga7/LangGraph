"""
react_pydantic.py: Agente ReAct con 2 tools utilizando Pydantic y un nodo evaluador.
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()


# 1. Definición del Esquema Pydantic para el resultado estructurado
class EvaluationResult(BaseModel):
    summary: str = Field(
        description="Resumen final o respuesta explicativa de la tarea realizada."
    )
    score: int = Field(
        description="Puntuación o confianza del resultado (1 a 10)."
    )
    is_successful: bool = Field(
        description="Indica si la ejecución de la tarea se completó exitosamente."
    )
    next_action: Literal["continue", "finish"] = Field(
        description="Determina si el agente necesita continuar razonando o dar por terminada la ejecución."
    )


# 2. Definición de Herramientas (Tools)
@tool
def multiply(a: float, b: float) -> float:
    """Multiplica dos números."""
    return a * b


@tool
def search_word_length(word: str) -> int:
    """Calcula la longitud de una palabra dada."""
    return len(word)


tools = [multiply, search_word_length]


# 3. Inicialización del modelo con Herramientas y Salida Estructurada
llm = ChatGroq(
    # modelos: qwen-2.5-coder-32b, llama-3.3-70b-versatile
    model="openai/gpt-oss-20b",
    temperature=0
)

# Unimos las herramientas al LLM base para las decisiones intermedias
llm_with_tools = llm.bind_tools(tools)

# Creamos la versión del LLM que fuerza la salida estructurada con EvaluationResult
llm_structured = llm.with_structured_output(EvaluationResult)


# 4. Definición de los Nodos del Agente ReAct
def agent_node(state: MessagesState):
    """
    Nodo ejecutor principal del agente. Genera mensajes e invoca herramientas si es necesario.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def evaluation_node(state: MessagesState):
    """
    Nodo de evaluación final que procesa la historia de mensajes 
    y la formatea según el esquema EvaluationResult usando Pydantic.
    """
    messages = state["messages"]
    prompt = [
        SystemMessage(
            content="Evalúa el resultado de la conversación y de las herramientas ejecutadas, "
                    "y proporciona el dictamen final estructurado."
        )
    ] + messages

    # Invocar usando el LLM formateado con structured output
    structured_response: EvaluationResult = llm_structured.invoke(prompt)

    # Retornar el objeto Pydantic envuelto en la respuesta
    return {"messages": [HumanMessage(content=structured_response.model_dump_json())]}


# 5. Condicional de Enrutamiento (Router)
def should_continue(state: MessagesState) -> Literal["tools", "evaluator"]:
    messages = state["messages"]
    last_message = messages[-1]

    # Si la última respuesta del modelo contiene tool_calls, enviamos a ejecutar las herramientas
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Si no requiere más herramientas, pasamos al nodo de estructuración y evaluación
    return "evaluator"


# 6. Construcción del Grafo LangGraph
workflow = StateGraph(MessagesState)

# Agregar nodos
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("evaluator", evaluation_node)

# Establecer punto de entrada
workflow.add_edge(START, "agent")

# Condicional de paso: agent -> tools OR evaluator
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "evaluator": "evaluator"
    }
)

# Transiciones
# Una vez ejecutadas las herramientas, vuelve al agente
workflow.add_edge("tools", "agent")
# Finaliza la ejecución tras estructurar el resultado
workflow.add_edge("evaluator", END)

# Compilar el Grafo
graph = workflow.compile()


# 7. Ejecución de Ejemplo
if __name__ == "__main__":
    inputs = {
        "messages": [
            HumanMessage(
                content="Multiplica 12.5 por 4 y calcula cuántas letras tiene la palabra 'LangGraph'."
            )
        ]
    }

    for chunk in graph.stream(inputs, stream_mode="values"):
        last_msg = chunk["messages"][-1]
        print(f"[{last_msg.type.upper()}]: {last_msg.content}")
