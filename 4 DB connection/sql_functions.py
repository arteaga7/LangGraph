"""sql_funciontions.py
SQL functions for IA agents"""
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from dotenv import load_dotenv
load_dotenv()

# Configuración PostgreSQL
DB_URI = (
    "postgresql://postgres:zmBWbmxaNwSGVBrP@db.uflkuitzcnrjyqsgbczm.supabase.co:5432/postgres"
)

# PostgreSQL Checkpointer
conn = psycopg.connect(DB_URI, autocommit=True)
memory = PostgresSaver(conn)
# Crea automáticamente las tablas necesarias
memory.setup()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)


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


def delete_messages(thread_id: str, message_ids: list[int]):
    if not message_ids:
        return
    with conn.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM messages
            WHERE thread_id = %s
            AND id = ANY(%s);
            """,
            (thread_id, message_ids)
        )


def delete_summary(thread_id: str):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM messages
            WHERE thread_id = %s
            AND role = 'summary';
            """,
            (thread_id,)
        )


def get_conversation(thread_id: str):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                id,
                thread_id,
                role,
                content,
                created_at
            FROM messages
            WHERE thread_id = %s
            ORDER BY created_at ASC, id ASC;
            """,
            (thread_id,)
        )
        return cursor.fetchall()


def summarizer_node(state):
    messages = state["messages"]
    # Buscar resumen existente
    previous_summary = None
    for message in messages:
        if getattr(message, "name", None) == "conversation_summary":
            previous_summary = message.content

    # Obtener mensajes normales
    normal_messages = [
        message
        for message in messages
        if getattr(message, "name", None)
        != "conversation_summary"
    ]

    # Solo resumir cuando existan 12 mensajes
    if len(normal_messages) < 12:
        return {}

    # Seleccionar los 6 mensajes MÁS ANTIGUOS
    messages_to_summarize = normal_messages[:6]

    # Construir texto para el LLM
    conversation_text = ""
    for message in messages_to_summarize:
        if isinstance(message, HumanMessage):
            role = "Usuario"
        else:
            role = "Asistente"
        conversation_text += (f"{role}: {message.content}\n")

    # Incluir resumen anterior
    if previous_summary:
        prompt = f"""
        Tenemos un resumen anterior de una conversación:
        RESUMEN ANTERIOR:
        {previous_summary}

        Ahora tenemos estos 6 mensajes antiguos nuevos:
        {conversation_text}

        Crea un nuevo resumen que combine:

        1. El resumen anterior.
        2. La información importante de estos 6 mensajes.
        3. Las preferencias del usuario.
        4. Las decisiones tomadas.
        5. Los datos importantes.
        6. El contexto necesario para continuar la conversación.

        No inventes información.

        Mantén el resumen compacto pero suficientemente
        informativo para que otro modelo pueda continuar
        la conversación correctamente.
        """
    else:
        prompt = f"""
        Resume los siguientes 6 mensajes de una conversación:
        {conversation_text}

        Conserva:
        - temas importantes;
        - preguntas realizadas;
        - respuestas relevantes;
        - preferencias del usuario;
        - decisiones;
        - datos importantes;
        - contexto necesario para continuar la conversación.

        No inventes información.

        El resumen debe ser compacto pero informativo.
        """

    # Generar resumen
    summary_response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Eres un sistema especializado "
                    "en resumir conversaciones."
                )
            ),
            HumanMessage(content=prompt)
        ]
    )
    new_summary = summary_response.content

    # Eliminar mensajes anteriores del estado
    messages_to_remove = [
        RemoveMessage(id=message.id)
        for message in messages_to_summarize
    ]

    # Si existe resumen anterior, eliminarlo
    if previous_summary:
        for message in messages:
            if getattr(message, "name", None) == "conversation_summary":
                messages_to_remove.append(
                    RemoveMessage(id=message.id)
                )

    # Crear nuevo mensaje de resumen
    summary_message = SystemMessage(
        content=new_summary,
        name="conversation_summary"
    )

    # Actualizar estado de LangGraph
    return {
        "messages": (messages_to_remove
                     + [summary_message]
                     )
    }
