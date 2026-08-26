"""
memory_postgresql.py
Agente con memoria persistente guardada en una base de datos PostgreSQL con LangGraph.
LangGraph crea tablas automaticamnte para guardar el estado, pero no es posible ver la conversacion.
Por lo tanto, se creara una tabla con las conversaciones.
"""
from pathlib import Path
from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
import psycopg

load_dotenv()

# Configuración PostgreSQL
DB_URI = (
    "postgresql://postgres:zmBWbmxaNwSGVBrP@db.uflkuitzcnrjyqsgbczm.supabase.co:5432/postgres"
)

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
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
# PostgreSQL Checkpointer
conn = psycopg.connect(DB_URI, autocommit=True)
memory = PostgresSaver(conn)
# Crea automáticamente las tablas necesarias
memory.setup()
graph = workflow.compile(checkpointer=memory)


# Crea tabla personalizada para las conversaciones
def create_messages_table():
    """Crea la tabla messages si todavía no existe."""
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)


# Guarda mensajes en tabla messages
def save_message(thread_id: str, role: str, content: str):
    """Guarda un mensaje en la tabla messages."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO messages (
                thread_id,
                role,
                content
            )
            VALUES (%s, %s, %s);
            """,
            (thread_id, role, content)
        )


def chat(message, thread_id="sesion_terminal"):
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
