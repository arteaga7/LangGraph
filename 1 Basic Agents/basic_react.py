"""
basic_react.py: Agente ReAct con 2 tools.
"""
# Importa utilidades para cargar variables de entorno desde un fichero .env
from dotenv import load_dotenv
# Decorador para registrar funciones Python como "tools" en LangChain
from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# Tipo de estado estándar: diccionario con clave "messages"
from langgraph.graph import MessagesState
# Nodo preconstruido de LangGraph para ejecutar tools
from langgraph.prebuilt import ToolNode
# Tipo de mensaje humano para iniciar la conversación
from langchain_core.messages import HumanMessage
# END: estado terminal; StateGraph: constructor de grafos
from langgraph.graph import END, StateGraph

load_dotenv()  # Carga variables de entorno


@tool  # Herramienta para realizar cálculos matemáticos
def calculator(expression: str) -> float:
    """
    Evalúa una expresión matemática simple. Debes usar esta herrammienta cuando el usuario te pide un cálculo matemático
    Ejemplo:
    "2 + 3 * 5"
    """
    return eval(expression)


@tool  # Herramienta para escribir en un fichero de texto
def write_text_file(path: str, content: str) -> str:
    """
    Escribe contenido en un archivo.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return "Archivo creado correctamente"


# Lista de herramientas disponibles
tools = [calculator, write_text_file]

llm = ChatGroq(
    # modelos: qwen-2.5-coder-32b, llama-3.3-70b-versatile
    model="openai/gpt-oss-20b",
    temperature=0
).bind_tools(tools)

SYSTEM_MESSAGE = """  
Eres un asistente capaz de utilizar herramientas para responder al usuario. Si encuentras una herramienta
útil para la solicitud, debes usarla antes de utilizar tu propio conocimiento.
"""


def run_agent_reasoning_engine(state: MessagesState) -> MessagesState:
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *
            state["messages"]]  # Sistema + historial de mensajes
    )

    # Devuelve el estado parcial: añade el nuevo mensaje generado por el LLM
    return {"messages": [response]}


# Nodo tools
# Crea el nodo que ejecutará herramientas cuando el LLM las solicite
tool_node = ToolNode(tools)


# Función de enrutado condicional: decide si el flujo termina o usa tools
def should_continue(state: dict) -> str:
    # Si el último mensaje NO contiene llamadas a herramientas...
    if not state["messages"][-1].tool_calls:
        return END  # ...termina el flujo (no hay acciones que ejecutar)
    return "act_tools"  # Si hay tool_calls, continúa hacia el nodo de acción


# Crea un grafo cuyo estado sigue el esquema MessagesState (usa clave "messages")
flow = StateGraph(MessagesState)

# Añade el nodo de razonamiento al grafo
flow.add_node("agent_reason", run_agent_reasoning_engine)

# Añade el nodo de acción (ToolNode) que ejecuta herramientas
flow.add_node("act_tools", tool_node)

flow.add_conditional_edges(  # Añade aristas condicionales que salen del nodo de razonamiento
    "agent_reason",  # Nodo origen (desde donde se decide)
    should_continue  # Función que inspecciona el estado y devuelve el "siguiente" destino
)

# Define que el flujo comience en el nodo de razonamiento
flow.set_entry_point("agent_reason")

# Tras ejecutar tools, vuelve al nodo de razonamiento (ciclo ReAct)
flow.add_edge("act_tools", "agent_reason")

graph = flow.compile()  # Compila el grafo en una aplicación ejecutable (Runnable)


if __name__ == "__main__":
    print("Hola agente ReAct")
    res = graph.invoke(  # Ejecuta el grafo pasando un estado inicial con un mensaje humano
        {
            "messages": [
                HumanMessage(
                    content="quiero que sumes 10 y 23, el resultado quiero que lo escribas en un fichero nuevo en la ruta '/home/ant/resultado.txt'"
                )
            ]
        }
    )
    # Imprime el contenido del último mensaje generado por el agente
    print(res["messages"][-1].content)
