"""
asyncronous.py
Grafo con nodos (B y B2) ejecutandose en paralelo con C.
"""
from pathlib import Path
# Importa el constructor de grafos de LangGraph y los nodos especiales START y END
from langgraph.graph import StateGraph, START, END
# TypedDict permite definir un diccionario con estructura tipada (clave y tipo de valor)
from typing_extensions import TypedDict
# Importa tipos: Annotated permite añadir metadata a un tipo y Any significa “cualquier tipo”
from typing import Annotated, Any
import operator  # Importa el módulo operator, aquí se usará operator.add para definir cómo se combinan los resultados del estado
# Importa la función que permite cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
load_dotenv()  # Ejecuta la carga de variables del archivo .env al entorno del proceso (útil para claves API u otras configuraciones)


class State(TypedDict):  # Define la estructura del estado que se moverá entre nodos del grafo
    # Campo del estado que es una lista y que se agregará usando operator.add cuando varios nodos escriban al mismo tiempo
    agregacion: Annotated[list, operator.add]


class ReturnNodeValue:  # Clase que define el comportamiento de un nodo del grafo
    # Constructor que recibe el valor que el nodo añadirá al estado
    def __init__(self, valor_insertado: str):
        # Guarda internamente el valor que el nodo devolverá cuando se ejecute
        self._value = valor_insertado

    # Permite que el objeto se comporte como una función cuando el grafo ejecute el nodo
    def __call__(self, state: State) -> Any:
        # Importa el módulo time dentro del método (solo se usa aquí)
        import time
        # Simula una tarea que tarda tiempo (por ejemplo una llamada a API o a un LLM)
        time.sleep(1)

        # Muestra por consola qué valor añade el nodo y el estado actual
        print(f"Añadir {self._value} to {state['agregacion']}")

        # Devuelve una actualización parcial del estado que se combinará con el estado global
        return {"agregacion": [self._value]}


builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("Soy A"))
builder.add_edge(START, "a")
builder.add_node("b", ReturnNodeValue("Soy B"))
builder.add_node("b2", ReturnNodeValue("Soy B2"))
builder.add_node("c", ReturnNodeValue("Soy C"))
builder.add_node("d", ReturnNodeValue("Soy D"))
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b2")
# Podemos definir las 2 aristas en una sola línea
builder.add_edge(["b2", "c"], "d")
builder.add_edge("d", END)
graph = builder.compile()

Path("./img").mkdir(parents=True, exist_ok=True)
graph.get_graph().draw_mermaid_png(output_file_path="./img/asyncronous.png")

if __name__ == "__main__":
    print("Hola grafo asincrono 2")
    graph.invoke({"agregacion": []}, {"configurable": {"thread_id": "106"}})
