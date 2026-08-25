"""
reflective_react.py
El nodo ReAct piensa y decide si usar herramientas o dar una respuesta borrador, mientras que
el nodo Evaluador analiza la respuesta generada (en la ultima iteracion) por el agente ReAct.
"""
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pathlib import Path
load_dotenv()


# ---------------------------------------------------------
# 1. HERRAMIENTAS Y MODELOS EN CASCADA
# ---------------------------------------------------------
@tool
def buscar_info_interna(query: str) -> str:
    """Busca datos internos de la empresa."""
    return f"Resultado de {query}: Operación exitosa."


@tool
def calculator(expression: str) -> str:
    """
    Evalúa una expresión matemática simple. Debes usar esta herrammienta cuando el usuario te pide un cálculo matemático
    Ejemplo:
    "2 ** 4 + 1"
    """
    return str(eval(expression))


tools = [buscar_info_interna, calculator]

# MODELO RÁPIDO: Se encarga del razonamiento repetitivo y uso de herramientas
fast_react_llm = ChatGroq(model="openai/gpt-oss-20b",
                          temperature=0).bind_tools(tools)

# MODELO ROBUSTO: Se encarga EXCLUSIVAMENTE de evaluar la calidad final con limite de tokens
robust_evaluator_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0,
                                max_tokens=250)     # 👈 Limita la respuesta del evaluador a ~250 tokens


# ---------------------------------------------------------
# 2. ESQUEMA PYDANTIC PARA EVALUACIÓN ESTRICTA
# ---------------------------------------------------------
class EvaluationResult(BaseModel):
    is_valid: bool = Field(
        description="True si la respuesta resuelve la petición del usuario de forma precisa, False si faltan datos o hay alucinaciones.")
    feedback: str = Field(
        description="Instrucción concreta de corrección si is_valid es False. Dejar vacío si es True.")


evaluator_chain = robust_evaluator_llm.with_structured_output(EvaluationResult)


# ---------------------------------------------------------
# 3. ESTADO DEL GRAFO
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    is_valid: bool
    user_query: str


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
    """Nodo Evaluador protegido contra consumo excesivo de tokens."""

    # 1. Extraer la consulta inicial del usuario
    user_query = next(
        (msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), "")

    # 2. Obtenemos el borrador del agente
    draft_message = str(state["messages"][-1].content)

    # 3. Truncar textos extremadamente largos en el contexto enviado (Hard Limit)
    # Ejemplo: Recortamos a ~1500 caracteres (~300 tokens de contexto)
    max_char_limit = 1500
    truncated_draft = draft_message[:max_char_limit] + \
        ("..." if len(draft_message) > max_char_limit else "")
    truncated_query = user_query[:500] + \
        ("..." if len(user_query) > 500 else "")

    # 4. Prompt ultracompacto para ahorrar tokens de entrada
    eval_prompt = (
        f"Pregunta: {truncated_query}\n"
        f"Respuesta a evaluar: {truncated_draft}\n\n"
        "Evalúa si la respuesta final responde correctamente a la pregunta. "
        "Considera que puede haber utilizado herramientas. "
        "No penalices la respuesta por ser breve si la pregunta es sencilla."
    )

    try:
        # Invocamos la cadena estructurada
        evaluation: EvaluationResult = evaluator_chain.invoke(
            [HumanMessage(content=eval_prompt)])

        if not evaluation.is_valid:
            feedback_msg = HumanMessage(
                # Truncamos también el feedback
                content=f"[Feedback Evaluador]: {evaluation.feedback[:300]}"
            )
            return {
                "messages": [feedback_msg],
                "is_valid": False,
                "iterations": state["iterations"] + 1
            }

        return {"is_valid": True}

    except Exception as e:
        # Fallback de seguridad: Si falla el parser por recortes de tokens u otro error,
        # asumimos la respuesta como no válida para no romper el flujo del Grafo.
        print(f"⚠️ Error en el evaluador: {e}")
        return {
            "messages": [
                HumanMessage(
                    content="[Feedback Evaluador]: No fue posible validar la respuesta."
                )
            ],
            "is_valid": False,
            "iterations": state["iterations"] + 1
        }

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
        return "__end__"
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
builder.add_conditional_edges("react", route_from_react,
                              {"tools": "tools", "evaluator": "evaluator"})

builder.add_edge("tools", "react")

# Borde que evalúa si el agente logró superar la revisión
builder.add_conditional_edges("evaluator", route_from_evaluator,
                              {"react": "react", END: END})

graph = builder.compile()
Path("./img").mkdir(parents=True, exist_ok=True)
graph.get_graph().draw_mermaid_png(output_file_path="./img/reflective_react.png")

if __name__ == "__main__":
    print("Hola agente Reflective+ReAct")
    inputs = {
        "messages": [
            HumanMessage(
                content="cuanto es 2 elevado a la 4, más 1?")
        ]
    }

    # Transmitimos los eventos para ver la secuencia Reason -> Act -> Observe
    for chunk in graph.stream(inputs, stream_mode="values"):
        last_message = chunk["messages"][-1]
        print(f"\n[{last_message.__class__.__name__}]:")
        # Si el modelo decidió hacer llamadas a herramientas (ReAct: Act)
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                print(
                    f"  🔧 Llamando a la herramienta: '{tool_call['name']}' con argumentos: {tool_call['args']}")
        else:
            print(f"  {last_message.content}")

    """
    result = graph.invoke(  # Ejecuta el grafo pasando un estado inicial con un mensaje humano
        inputs
    )
    # Imprime el contenido del último mensaje generado por el agente
    #print(result["messages"][-1].content)
    """
