"""
hil_memorySaver.py
Persistencia (checkpointing) en LangGraph usando memoria RAM (MemorySaver) y HIL sin LLM.
Este script crea un grafo muy simple con tres nodos:
  START -> Paso_1 -> human_feedback (HIL) -> Paso_3 -> END

La clave para explicar *persistencia* aquí es el "checkpointer":
- MemorySaver guarda el estado del grafo en memoria (se pierde al cerrar el proceso).
- Además, `interrupt_before=["human_feedback"]` detiene la ejecución justo antes de pedir feedback humano,
  permitiendo inspeccionar/actualizar el estado y luego reanudar.

"""
# 1. Importaciones
# ----------------
# Importa el checkpointer en memoria (persistencia en RAM).
from langgraph.checkpoint.memory import MemorySaver
# Importa el constructor de grafos y los nodos especiales START/END.
from langgraph.graph import StateGraph, START, END
# Importa TypedDict para tipar el estado (diccionario con claves conocidas).
from typing import TypedDict
# Importa `load_dotenv` para cargar variables de entorno desde un archivo .env.
from dotenv import load_dotenv
# Carga automáticamente variables de entorno definidas en .env al entorno del proceso.
load_dotenv()


# Define la estructura del estado que circulará por el grafo.
class State(TypedDict):
    # Campo del estado para el input inicial (por ejemplo, lo que escribe el usuario).
    input: str
    # Campo del estado para guardar el feedback humano (se añadirá más tarde).
    user_feedback: str


# Define el primer nodo; recibe el estado actual y no devuelve nada (solo side-effects).
def Paso_1(state: State) -> None:
    # Muestra por consola que el grafo está ejecutando el nodo Paso_1.
    print("---Paso 1---")


# Define el nodo donde conceptualmente se pide feedback humano.
def human_feedback(state: State) -> None:
    # Muestra por consola que se llegó al punto de feedback (antes de interrumpir).
    print("---Intervención_humana---")


# Define el tercer nodo del grafo, para continuar después del feedback.
def Paso_3(state: State) -> None:
    # Muestra por consola que el grafo está ejecutando el nodo Paso_3.
    print("---Paso 3--")


# Crea un "builder" de grafo tipado con la estructura State.
builder = StateGraph(State)

# Registra el nodo llamado "Paso_1" y lo asocia a la función Paso_1.
builder.add_node("Paso_1", Paso_1)
# Registra el nodo "human_feedback" asociado a la función homónima.
builder.add_node("Intervención_humana", human_feedback)
# Registra el nodo llamado "Paso_3" y lo asocia a la función Paso_3.
builder.add_node("Paso_3", Paso_3)

# Define que la ejecución comienza en START y pasa a "Paso_1".
builder.add_edge(START, "Paso_1")
# Tras "Paso_1", la ejecución continúa hacia "human_feedback".
builder.add_edge("Paso_1", "Intervención_humana")
# Tras "human_feedback", la ejecución continúa hacia "Paso_3".
builder.add_edge("Intervención_humana", "Paso_3")
builder.add_edge("Paso_3", END)  # Tras "Paso_3", la ejecución termina en END.

# Crea el checkpointer: guardará snapshots del estado en memoria (persistencia temporal en RAM).
memory = MemorySaver()

graph = builder.compile(  # Compila el builder en un grafo ejecutable.
    # Activa el checkpointing usando MemorySaver (para poder recuperar/actualizar estado).
    checkpointer=memory,
    # Interrumpe la ejecución *antes* de entrar en "human_feedback".
    interrupt_before=["Intervención_humana"],
)

# Exporta un PNG del grafo (útil para doc/explicación).
graph.get_graph().draw_mermaid_png(output_file_path="./3 Memory/graph_hil.png")

if __name__ == "__main__":
    # Define un "thread_id" para aislar la sesión/ejecución del grafo.
    thread = {"configurable": {"thread_id": "577"}}

    # Define el estado inicial: solo aporta la clave "input".
    initial_input = {"input": "Hola LangGraph"}

    # Ejecuta el grafo en modo streaming (va emitiendo eventos) con el input.
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        # Imprime cada "evento" (estado) emitido durante la ejecución (en modo values: solo el estado actual sin metadatos).
        print(event)

    # Muestra cuál es el siguiente nodo pendiente (debería ser "Intervención_humana").
    print(graph.get_state(thread).next)

    # Pide al usuario el feedback para guardar en estado.
    user_input = input("Dime cómo quieres actualizar el estado: ")

    graph.update_state(  # Actualiza el estado persistido en el checkpointer para este thread.
        thread,  # Indica qué sesión/hilo (thread_id) se actualiza.
        # Rellena la clave "user_feedback" del estado.
        {"user_feedback": user_input},
        # Registra que esta actualización corresponde al paso "Intervención_humana".
        as_node="Intervención_humana",
    )

    # Mensaje informativo por consola.
    print("--Estado después de actualizar--")
    # Imprime el estado completo tras la actualización (incluye input y user_feedback).
    print(graph.get_state(thread))

    # Vuelve a mostrar el siguiente nodo pendiente (ahora debería ser "Paso_3").
    print(graph.get_state(thread).next)

    # Reanuda la ejecución desde el checkpoint (None = sin input nuevo).
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)  # Imprime los eventos restantes hasta END.
