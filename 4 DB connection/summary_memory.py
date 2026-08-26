"""
summary_memory.py
Agente con memoria persistente guardada en una base de datos PostgreSQL con LangGraph.
LangGraph crea tablas automaticamnte para guardar el estado, pero no es posible ver la conversacion.
Por lo tanto, se creara una tabla con las conversaciones. Cada 6 mensajes se hace un resumen y se
reemplaza en las tablas.
"""
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from sql_functions import *
from dotenv import load_dotenv
load_dotenv()

workflow = StateGraph(state_schema=MessagesState)


def chatbot_node(state):
    """Nodo que procesa mensajes y genera respuestas."""
    system_prompt = """Eres un asistente amigable que recuerda
    conversaciones previas.
    Si existe un resumen de la conversación,
    utilízalo como contexto.
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow.add_node("chatbot", chatbot_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", "summarizer")
workflow.add_edge("summarizer", END)
graph = workflow.compile(checkpointer=memory)


def chat(message, thread_id="sesion_terminal"):
    config = {"configurable": {"thread_id": thread_id}}
    save_message(thread_id=thread_id, role="user", content=message)
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )

    # Buscar específicamente la respuesta AI
    assistant_messages = [
        message
        for message in result["messages"]
        if message.type == "ai"
    ]
    response = assistant_messages[-1].content
    save_message(thread_id=thread_id, role="assistant", content=response)

    # COMPROBAR SI SE PRODUJO UN RESUMEN
    summary_messages = [
        message
        for message in result["messages"]
        if getattr(message, "name", None)
        == "conversation_summary"
    ]

    if summary_messages:
        summary = summary_messages[-1]
        # Obtener registros actuales
        conversation = get_conversation(thread_id)
        # ----------------------------------------------------
        # Los primeros mensajes normales que fueron
        # resumidos deben eliminarse de la tabla.
        # Conservamos el summary anterior si existe
        # y reemplazamos los mensajes resumidos.
        # ----------------------------------------------------
        normal_rows = [
            row
            for row in conversation
            if row[2] != "summary"
        ]

        # Eliminar los 6 mensajes más antiguos
        if len(normal_rows) >= 6:
            rows_to_delete = normal_rows[:6]
            ids_to_delete = [
                row[0]
                for row in rows_to_delete
            ]
            delete_messages(
                thread_id=thread_id,
                message_ids=ids_to_delete
            )

        # Eliminar resumen anterior de la tabla
        delete_summary(thread_id)

        # Guardar nuevo resumen
        save_message(thread_id=thread_id, role="summary",
                     content=summary.content)
    return response


if __name__ == "__main__":
    create_messages_table()
    print("Chat en terminal (escribe 'salir' para terminar)\n")
    session_id = "sesion_1"
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
