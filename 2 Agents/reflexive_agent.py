"""
reflexive_agent.py
Agente de reflexión para generar y evaluar una consulta SQL mediante un proceso iterativo de reflexión.
Este nodo son 2 nodos realmente. El nodo 'generator_node' genera el texto y el nodo 'evaluator_node'
evalua el texto (la salida) del primer nodo.
"""
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

# 1. Definir el modelo
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


# 2. Definir el Estado del Grafo
class AgentState(TypedDict):
    task: str                 # La solicitud inicial del usuario
    draft: str                # La propuesta actual de respuesta/código
    feedback: str             # Crítica o correcciones sugeridas por el evaluador
    is_satisfactory: bool     # Flag para saber si se aprobó
    iterations: int           # Control para evitar bucles infinitos


# Schema Pydantic para forzar salida estructurada del Evaluador
class Evaluation(BaseModel):
    is_satisfactory: bool = Field(
        description="True si la respuesta cumple todas las reglas, False si no.")
    feedback: str = Field(
        description="Explicación detallada de los errores o sugerencias de mejora si is_satisfactory es False.")


# 3. Nodo Generador / Optimizador
def generator_node(state: AgentState) -> dict:
    task = state["task"]
    draft = state.get("draft", "")
    feedback = state.get("feedback", "")
    iterations = state.get("iterations", 0)

    if not draft:
        # Generación inicial
        prompt = f"""
        Genera una consulta SQL PostgreSQL para responder a esta solicitud:
        {task}
        Devuelve únicamente el código SQL.
        """
    else:
        # Optimización basada en retroalimentación
        prompt = (
            f"Tarea original:\n{task}\n\n"
            f"Tu borrador previo de SQL fue:\n{draft}\n\n"
            f"Retroalimentación recibida:\n{feedback}\n\n"
            f"Corrige la consulta SQL para solucionar los problemas indicados."
        )

    messages = [
        SystemMessage(
            content="""
            Eres un experto en PostgreSQL.
            Genera únicamente código SQL válido.
            No inventes columnas o tablas si el schema proporcionado no contiene la información necesaria.
            Si el schema no es suficiente para generar una consulta confiable, indícalo claramente.
            """
        ),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    return {
        "draft": response.content.strip(),
        "iterations": iterations + 1
    }


# 4. Nodo Evaluador
def evaluator_node(state: AgentState) -> dict:
    task = state["task"]
    draft = state["draft"]

    evaluator_llm = llm.with_structured_output(Evaluation)

    prompt = f"""
    Evalúa el siguiente código SQL para la tarea: "{task}"
    
    Código SQL a revisar:
    {draft}
    
    Reglas de evaluación:
    1. Debe usar sintaxis válida de PostgreSQL.
    2. Debe usar alias claros en los JOINs.
    3. NUNCA debe usar 'SELECT *', siempre debe explicitar las columnas.
    4. Si utiliza funciones de agregación junto con columnas no agregadas,
    debe incluir un GROUP BY explícito para esas columnas.
    """

    evaluation: Evaluation = evaluator_llm.invoke([
        SystemMessage(content="Eres un revisor estricto de código SQL."),
        HumanMessage(content=prompt)
    ])

    return {
        "is_satisfactory": evaluation.is_satisfactory,
        "feedback": evaluation.feedback
    }


# 5. Función condicional (Router)
def route_evaluation(state: AgentState) -> Literal["generator", "__end__"]:
    # Límite de seguridad de 3 iteraciones para evitar ciclos infinitos
    if state["is_satisfactory"] or state["iterations"] >= 3:
        return "__end__"
    return "generator"


# 6. Construir el Grafo de LangGraph
workflow = StateGraph(AgentState)

# Agregar nodos
workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)

# Agregar bordes/flujos
workflow.add_edge(START, "generator")
workflow.add_edge("generator", "evaluator")

# Borde condicional desde el evaluador
workflow.add_conditional_edges(
    "evaluator",
    route_evaluation,
    {
        "generator": "generator",
        "__end__": END
    }
)

graph = workflow.compile()
graph.get_graph().draw_mermaid_png(output_file_path="./img/reflexive_agent.png")

# 7. Ejecución de prueba
if __name__ == "__main__":
    initial_state = {
        "task": "Obtén el total de ventas y el nombre del cliente para los clientes que hayan comprado más de $1,000 en 2024.",
        "draft": "",
        "feedback": "",
        "is_satisfactory": False,
        "iterations": 0
    }

    print("--- INICIANDO FLUJO REFLEXIVO ---")
    for event in graph.stream(initial_state):
        for node, values in event.items():
            print(f"\n[Nodo ejecutado: {node}]")
            if node == "generator":
                print(
                    f"Borrador (Iteración {values.get('iterations')}):\n{values.get('draft')}")
            elif node == "evaluator":
                print(f"¿Aprobado?: {values.get('is_satisfactory')}")
                print(f"Feedback: {values.get('feedback')}")
