import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


@tool
def ejecutar_consulta_sql(query: str) -> str:
    """Ejecuta una consulta SQL en la base de datos de producción y devuelve los resultados."""
    # Simulación de respuesta de base de datos
    if "ventas" in query.lower():
        return "Resultados SQL: [{'cliente': 'TechCorp', 'total_2024': 15400.00}, {'cliente': 'DataSoft', 'total_2024': 8200.50}]"
    return "Resultados SQL: []"


@tool
def calcular_descuento(monto: float, porcentaje: float) -> str:
    """Calcula el descuento aplicable a un monto dado."""
    monto_final = monto * (1 - porcentaje / 100)
    return f"Monto original: ${monto:.2f}, Con {porcentaje}% desc: ${monto_final:.2f}"


tools = [ejecutar_consulta_sql, calcular_descuento]


# 2. Configurar el LLM con bindings para las Herramientas
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
).bind_tools(tools)


# 3. Definir el Estado del Grafo
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 4. Nodo Agente (Razonamiento / Decisiones)
def agent_node(state: AgentState):
    system_prompt = SystemMessage(
        content="Eres un asistente analista de datos. Usa las herramientas disponibles para responder a las consultas."
    )
    # Incluimos el system prompt junto con el historial de mensajes
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# 5. Construir el Grafo ReAct
builder = StateGraph(AgentState)

# Agregar el nodo del agente y el nodo preconstruido de herramientas
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

# Definir la entrada
builder.add_edge(START, "agent")

# Condición: Si el agente decide llamar una herramienta va a "tools", de lo contrario termina (END)
builder.add_conditional_edges(
    "agent",
    tools_condition,  # Función preconstruida de LangGraph para evaluar tool_calls
)

# Tras ejecutar la herramienta, siempre regresamos al agente para que interprete el resultado
builder.add_edge("tools", "agent")

app = builder.compile()


# 6. Ejemplo de ejecución
if __name__ == "__main__":
    print("--- INICIANDO AGENTE REACT ---")

    inputs = {
        "messages": [
            HumanMessage(
                content="¿Cuál fue el total de ventas de TechCorp en 2024 y cuánto sería si les aplicamos un 10% de descuento?")
        ]
    }

    # Transmitimos los eventos para ver la secuencia Reason -> Act -> Observe
    for chunk in app.stream(inputs, stream_mode="values"):
        last_message = chunk["messages"][-1]
        print(f"\n[{last_message.__class__.__name__}]:")

        # Si el modelo decidió hacer llamadas a herramientas (ReAct: Act)
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                print(
                    f"  🔧 Llamando a la herramienta: '{tool_call['name']}' con argumentos: {tool_call['args']}")
        else:
            print(f"  {last_message.content}")
