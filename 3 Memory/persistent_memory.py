"""
persistent_memory.py
Agente simple con memoria persistente usando Sqlite en LangGraph.
"""
from pathlib import Path
from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
load_dotenv()

SQLITE_DB = "./3 Memory/historial.db"
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)

workflow = StateGraph(state_schema=MessagesState)


def chatbot_node(state):
    """Nodo que procesa mensajes y genera respuestas."""
    system_prompt = "Eres un asistente amigable que recuerda conversaciones previas."
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")

# Compilar el grafo
conn = sqlite3.connect(SQLITE_DB, check_same_thread=False)
memory = SqliteSaver(conn)
graph = workflow.compile(checkpointer=memory)
# Path("./img").mkdir(parents=True, exist_ok=True)
# graph.get_graph().draw_mermaid_png(output_file_path="./img/persistent_memory.png")


def chat(message, thread_id="sesion_terminal"):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]}, config)
    return result["messages"][-1].content


if __name__ == "__main__":
    print("Chat en terminal (escribe 'salir' para terminar)\n")
    session_id = "sesion_2"

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego!")
            break

        respuesta = chat(user_input, session_id)
        print("Asistente:", respuesta)
