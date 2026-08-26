"""
sliding_memory.py
Agente simple con ventana de memoria deslizante (en RAM con MemorySaver) en LangGraph.
"""
from pathlib import Path
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.messages import trim_messages
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
workflow = StateGraph(state_schema=MessagesState)

trimmer = trim_messages(
    strategy="last",
    max_tokens=4,                   # Maximum messages
    token_counter=len,
    start_on="human",
    include_system=True
)


def chatbot_node(state):
    """Nodo que procesa mensajes y genera respuestas."""
    trimmed_messages = trimmer.invoke(state["messages"])
    system_prompt = "Eres un asistente amigable que recuerda conversaciones previas."
    # For simple memory use 'state["messages"]' insted of 'trimmed_messages'
    messages = [SystemMessage(content=system_prompt)] + trimmed_messages
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")

# Compilar el grafo
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
Path("./img").mkdir(parents=True, exist_ok=True)
graph.get_graph().draw_mermaid_png(output_file_path="./img/sliding_memory.png")


def chat(message, thread_id="sesion_terminal"):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]}, config)
    return result["messages"][-1].content


if __name__ == "__main__":
    print("Chat en terminal (escribe 'salir' para terminar)\n")
    session_id = "sesion_terminal"

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
