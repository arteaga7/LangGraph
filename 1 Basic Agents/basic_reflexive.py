"""
basic_reflexive.py
Agente de reflexión para generar y evaluar publicaciones de LinkedIn
mediante un proceso iterativo de reflexión. Este nodo son 2 nodos realmente.
El nodo generation genera el texto y el nodo reflection evalua el texto del primer nodo.
"""
from typing import TypedDict, Annotated
from dotenv import load_dotenv
# BaseMessage es la clase base para mensajes; HumanMessage representa mensajes del usuario
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Reductor que fusiona listas de mensajes (añade en lugar de reemplazar)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# Para importar con Ollama
# from langchain_ollama import ChatOllama

load_dotenv()

# Plantilla de prompt para la fase de generación: el modelo crea y mejora publicaciones
generation_prompt = ChatPromptTemplate.from_messages(
    [
        # Mensaje de sistema: define al modelo como asistente que escribe publicaciones
        (
            "system",
            "Eres un asistente de influencer tecnológico de LinkedIn encargado de escribir excelentes publicaciones. "
            "Genera la mejor publicación posible según la petición del usuario. "
            "Si el usuario aporta crítica, responde con una versión revisada de tus intentos anteriores.",
        ),
        # Marcador para el historial de mensajes
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Plantilla de prompt para la fase de reflexión: el modelo actúa como evaluador
# y genera crítica y recomendaciones sobre la publicación del usuario
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        # Mensaje de sistema: define el rol y comportamiento del modelo
        (
            "system",
            "Eres un influencer viral de LinkedIn que evalúa publicaciones. Genera crítica y recomendaciones para la publicación del usuario. "
            "Siempre proporciona recomendaciones detalladas, incluyendo aspectos como longitud, viralidad, estilo, etc.",
        ),
        # Marcador que se reemplaza por el historial de mensajes de la conversación
        # (petición del usuario, borradores previos, críticas, etc.)
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatGroq(
    # modelos: llama-3.1-8b-instant, qwen-2.5-coder-32b, llama-3.3-70b-versatile
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Cadena de generación: combina el prompt de generación con el LLM mediante el operador |
# Flujo: generation_prompt formatea los mensajes → llm genera la respuesta
generate_chain = generation_prompt | llm

# Cadena de reflexión: combina el prompt de reflexión con el LLM
# Flujo: reflection_prompt formatea los mensajes → llm genera crítica y recomendaciones
reflect_chain = reflection_prompt | llm

# Esquema del estado del grafo: un diccionario con clave "messages"
# Annotated con add_messages hace que los mensajes se acumulen en lugar de sustituirse


# Estado del grafo
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Nodo de generación: invoca la cadena de generación con el historial actual
# y devuelve el estado actualizado con la nueva publicación generada
def generation_node(state: MessageGraph):
    # Invocamos la cadena de generación (prompt + LLM) con el historial
    generated_response = generate_chain.invoke({"messages": state["messages"]})
    # Devolvemos el estado actualizado; add_messages fusionará la respuesta al historial
    return {"messages": [generated_response]}


# Nodo de reflexión: invoca la cadena de reflexión que critica la publicación
# y devuelve la crítica como mensaje humano para que el siguiente ciclo la use
def reflection_node(state: MessageGraph):
    # Invocamos la cadena de reflexión con todo el historial
    res = reflect_chain.invoke({"messages": state["messages"]})
    # Envolvemos la respuesta en HumanMessage para simular feedback de humano/externo
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)

GENERATE = "generate"  # Nodo que genera o mejora la publicación
REFLECT = "reflect"    # Nodo que evalúa y critica la publicación

# Añadimos los dos nodos al grafo especificando la función que se ejecutará en cada nodo
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)


# Función de decisión: determina si continuar iterando o finalizar
# Limita las iteraciones para evitar bucles infinitos (ejemplo:máx. ~3 ciclos con 6 mensajes)
# Cada ciclo tiene 2 mensajes (de IA y de Human)
def should_continue(state: MessageGraph):
    # Si hay más de 6 mensajes, damos por finalizado el proceso
    if len(state["messages"]) > 4:
        return END
    # Si no, pasamos al nodo de reflexión para otra ronda de mejora
    return REFLECT


# Arista condicional desde GENERATE: según should_continue va a END o a REFLECT
builder.add_conditional_edges(GENERATE, should_continue)

# Arista fija: tras reflexionar, siempre volvemos a generar
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

if __name__ == "__main__":
    print("\n--- Ejecutando Agente Reflexivo ---")

    inputs = {
        "messages": [
            HumanMessage(
                content="""
Mejorar esta publicación de LinkedIn:
¡La nueva funcionalidad Tool Calling de @LangChainAI es realmente una revolución!
Después de mucha espera, por fin está disponible y facilita enormemente la implementación de agentes en diferentes modelos gracias a la integración con function calling.
¿Ya la probaste? Cuéntame tu experiencia en los comentarios.
""".strip()
            )
        ]
    }

    # Invocamos el grafo completo con la entrada
    response = graph.invoke(inputs)

    # Impresión limpia del resultado final
    print("\n--- Historial Completo de Mensajes ---")
    for i, msg in enumerate(response["messages"]):
        role = "HUMANO/FEEDBACK" if isinstance(msg, HumanMessage) else "AGENTE"
        print(f"\n[{i+1}] {role}:\n{msg.content}")
