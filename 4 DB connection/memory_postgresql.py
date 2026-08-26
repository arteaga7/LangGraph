"""
memory_postgresql.py
Agente con memoria persistente guardada en una base de datos PostgreSQL con LangGraph.
LangGraph crea tablas automaticamnte para guardar el estado, pero no es posible ver la conversacion.
Por lo tanto, se creara una tabla con las conversaciones.
"""
from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from sql_functions import *
from dotenv import load_dotenv
load_dotenv()

workflow = StateGraph(state_schema=MessagesState)


def chatbot_node(state):
    """Nodo que procesa mensajes y genera respuestas."""
    system_prompt = (
        "Eres un asistente amigable que recuerda conversaciones previas."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
# 'memory' imported from sql_funciontions.py
graph = workflow.compile(checkpointer=memory)


def chat_memory(message, thread_id="sesion_terminal"):
    config = {"configurable": {"thread_id": thread_id}}
    save_message(thread_id=thread_id, role="user", content=message)
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    response = result["messages"][-1].content
    save_message(thread_id=thread_id, role="assistant", content=response)
    return response


if __name__ == "__main__":
    create_messages_table()
    print("Chat en terminal (escribe 'salir' para terminar)\n")
    session_id = "sesion_y"
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

        respuesta = chat_memory(user_input, session_id)
        print("Asistente:", respuesta)
