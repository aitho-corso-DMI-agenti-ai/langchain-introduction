import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from dotenv import load_dotenv

    from langchain_openai import ChatOpenAI
    from langchain_mistralai import ChatMistralAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
    )
    from langchain_core.messages import AIMessage

    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain.memory.chat_message_histories import ChatMessageHistory

    from collections import defaultdict

    load_dotenv()
    return (
        AIMessage,
        ChatMessageHistory,
        ChatMistralAI,
        ChatOpenAI,
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
        RunnableWithMessageHistory,
        StrOutputParser,
        defaultdict,
        mo,
        os,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Introduzione a LangChain e LangGraph""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In questo notebook di introduzione, vedremo come utilizzare prima LangChain e dopo LangGraph per la creazione di un agente con memoria conversazionale.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. LangChain - Approccio Legacy""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Cominciamo implementando una semplice `chain` con memoria conversazionale. Per ogni messaggio, l'agente dovrà mantenere una propria memoria e ricordare la storia dell'intera conversazione.
    In questo contesto, stiamo aumentando le capacità del LLM, fornendogli la capacità di ricordare tutto ciò che gli viene detto, e di riutilizzare le informazioni mantenute per soddisfare altre richieste nel corso della conversazione.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""È anzitutto necessario creare un template, in maniera tale che il modello conversazionale possa tener conto della cronologia della conversazione fatta, sotto forma di messaggi.""")
    return


@app.cell
def _(ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder):
    prompt = ChatPromptTemplate(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    return (prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'oggetto `ChatPromptTemplate` permette di costruire un prompt che può contenere più messaggi, non solo uno.

    In particolare:

    - `MessagePlaceholder(variable_name="chat_history")`: è un "segnaposto" che verrà sostituito in fase di esecuzione con la cronologia della chat (cioè una lista di messaggi passati). LangChain andrà quindi a sostituire `{chat_history}` con tutti i messaggi precedenti della conversazione, sia utente che assistente, nella seguente maniera:
        ```python
        chat_history = [
            HumanMessage(content="Ciao!"),
            AIMessage(content="Ciao! Come posso aiutarti?"),
        ]
        ```
    - `HumanMessagePromptTemplate.from_template("{text}")`: è un template per un messaggio dell’utente. `{text}` è un segnaposto che sarà rimpiazzato con l'input attuale dell'utente (una stringa), come una domanda o comando.

    Di seguito un esempio concreto di sostituzione automatica.
    """
    )
    return


@app.cell
def _(
    AIMessage,
    ChatPromptTemplate,
    HumanMessage,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    prompt,
):
    # Esempio di storico di una conversazione
    chat_history_example = [
        HumanMessage(content="Ciao!"),
        AIMessage(content="Ciao! Come posso aiutarti?")
    ]

    # Esempio di query utente
    text_example = "Parlami dell'università di Catania"

    # Esempio di prompt template
    prompt_example = ChatPromptTemplate(
        [
            MessagesPlaceholder(variable_name="chat_history_example"),
            HumanMessagePromptTemplate.from_template("{text_example}"),
        ]
    )

    # Formattazione e visualizzazione del prompt
    formatted_prompt = prompt.format_messages(
        chat_history=chat_history_example,
        text=text_example
    )

    print(formatted_prompt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Creiamo adesso una `chain` conversazionale.

    Il modello che utilizzeremo è `mistral-small-latest` di Mistral. Ci si può iscrivere al piano free di Mistral e richiedere una API Key seguendo questi passi:

    - creare un account Mistral (o effettuare il login, se già iscritti) a questo [link](https://mistral.ai/);
    - recarsi poi a questo [link](https://console.mistral.ai/home), navigare su *Spazio di lavoro* (nella toolbar a sinistra) e selezionare *Chiavi API*;
    - cliccare su *Scegli un piano* e selezionare il piano **Experiment**;
    - verificare utilizzando il proprio numero di cellulare;
    - dopo la verifica, generare una nuova chiave cliccando su *Crea nuova chiave*;
    - copiare la chiave e incollarla nel file `.env` sotto la variabile `MISTRAL_API_KEY`.
    """
    )
    return


@app.cell
def _(ChatMistralAI, StrOutputParser, os, prompt):
    llm = ChatMistralAI(
        name="mistral-small-latest",
        temperature=0.7,
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    chain = prompt | llm | StrOutputParser()
    return chain, llm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    La catena `chain` è costituita da tre blocchi: prompt → LLM → output parser. In particolare:

    - `prompt`: il template visto prima.
    - `llm`: l'effettivo modello LLM, il "motore" dell'intera catena conversazionale.
    - `StrOutputParser()`: converte l'output del modello (che di default è un oggetto strutturato) in una semplice stringa, pronta da usare o stampare.

    Mettiamo adesso insieme tutti gli elementi visti finora.
    """
    )
    return


@app.cell
def _(ChatMessageHistory, RunnableWithMessageHistory, chain, defaultdict):
    store = defaultdict(ChatMessageHistory)

    # chain con memoria messaggi
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        # questa è l'effettiva memoria messaggi!
        lambda session_id: store[session_id],
        input_messages_key="text",
        history_messages_key="chat_history",
    )
    return (chain_with_memory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    L'oggetto `RunnableWithMessageHistory` combina la `chain` (prompt → LLM → output parser) con un meccanismo di memoria automatica per tenere traccia della cronologia chat per ogni sessione o utente separatamente.

    Analizziamone i parametri:

    - `chain`: è la pipeline di conversazione definita prima.
    - `lambda session_id: store[session_id]`: tramite essa si accede ad una memoria specifica per ogni utente o sessione, usando una chiave `session_id`. In particolare `store` che per ogni chiave contiene lo storico di conversazione di uno specifico utente.
    - `input_messages_key="text"`: stiamo dicendo dicendo che il messaggio dell'utente si trova nella chiave `text`; questo si ricollega a `{text}` nel `HumanMessagePromptTemplate`.
    - Stiamo dicendo di generare il prompt inserendo la cronologia sotto il nome `chat_history`, che è lo stesso del `MessagesPlaceholder`.

    Definiamo adesso una funzione helper per invocare la `chain`.
    """
    )
    return


@app.cell
def _(chain_with_memory):
    def invoke_memory_chain(message):
        # Forza a stringa in tutti i casi
        message_str = str(getattr(message, "content", message))

        response = chain_with_memory.invoke(
            {"text": message_str},
            config={"configurable": {"session_id": "default"}}
        )
        return response

    return (invoke_memory_chain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Invochiamo infine la `chain`.""")
    return


@app.cell(hide_code=True)
def _(invoke_memory_chain, mo):
    mo.ui.chat(
        invoke_memory_chain,
        prompts=["Ciao", "Come stai?"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Abbiamo appena creato una `chain` in grado di mantenere memoria della conversazione!""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. LangGraph - Approccio moderno""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.1 Agente ReAct con memoria""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Il nome ReAct viene da due concetti fondamentali: **Re**asoning + **Act**ing.

    Questi agenti pensano, decidono se usare strumenti, li usano, riflettono e agiscono di nuovo — finché trovano una risposta soddisfacente!

    Un agente ReAct è un agente multi-step che:

    - **Ragiona**: analizza l'input, valuta cosa sa e cosa gli manca.
    - **Agisce**: se serve, usa uno strumento (es: calcolatrice, web search, DB).
    - **Osserva**: legge l’output dello strumento.
    - **Ragiona di nuovo**: aggiorna la sua strategia e decide se ha una risposta oppure continua.
    - Ripete il ciclo fino al "**Final Answer**".

    Utilizzando il più moderno framework LangGraph, reimplementiamo la chain vista prima sotto forma di Agente in grado di mantenere la memoria.
    """
    )
    return


@app.cell
def _():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
    from langchain.agents import Tool
    from langchain_core.messages import HumanMessage

    import wikipedia
    return HumanMessage, MemorySaver, Tool, create_react_agent, wikipedia


@app.cell
def _(MemorySaver):
    memory = MemorySaver()
    return (memory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In questo caso, la memoria è costituita da un `MemorySaver`, un oggetto in grado di salvare lo stato interno dell’agente a ogni passo del ragionamento.

    Quando si costruiscono agenti complessi con LangGraph (specialmente agenti ReAct), ogni "pensiero" e azione viene eseguito in una catena di passi. Se un passo fallisce o se vuoi riprendere la conversazione da dove si era interrotta, il checkpoint torna utile.

    Esempi di cosa salva:

    - Lo stato dell'agente (dati interni, memoria conversazionale, variabili temporanee)
    - Le azioni passate (tool usati, input/output)
    - Le risposte intermedie del modello (ragionamenti tipo "Penso che..." / "Devo cercare...")

    Ovviamente, questo tipo di memoria non è persistente; infatti, essa verrà persa una volta chiusa la sessione. Per avere una persistenza di memoria si potrebbero usare alternative come `FileSaver`, `SQLiteSaver` o `RedisSaver`.

    Creiamo adesso l'agent ReAct.
    """
    )
    return


@app.cell
def _(create_react_agent, llm, memory):
    react_agent = create_react_agent(
        model=llm,
        checkpointer=memory,
        tools=[]
    )
    return (react_agent,)


@app.cell
def _(HumanMessage, react_agent):
    def invoke_react_agent(message, debug=False):
        # Estrarre il testo nel modo più sicuro possibile
        text = getattr(message, "content", message)[-1].content

        response = react_agent.invoke(
            {"messages": HumanMessage(content=text)},
            config={
                "configurable": {
                    "session_id": "default",
                    "thread_id": "default",
                    "recursion_limit": 1,
                }
            },
            debug=debug,
        )
        return response["messages"][-1].content
    return (invoke_react_agent,)


@app.cell(hide_code=True)
def _(invoke_react_agent, mo):
    mo.ui.chat(
        invoke_react_agent,
        prompts=["Ciao!", "Come stai?"]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.2 Agente basato su Tool""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Aggiungiamo adesso un tool che possa essere utilizzato dall'agente.

    A scopo didattico, creiamo un tool che utilizzi la libreria `wikipedia` di python, per effettuare delle ricerche su Wikipedia. Il risultato, sarà un agente in grado di ricercare direttamente su wikipedia il necessario, qualora l'informazione richiesta non fosse nota.

    Innanzitutto, implementiamo una funzione `wikipedia_search` per effettuare una ricerca su Wikipedia.
    """
    )
    return


@app.cell
def _(wikipedia):
    def wikipedia_search(query: str):
        try:
            result = wikipedia.summary(query)
            return result
        except Exception as e:
            print(f"Errore nella ricerca: {e}")
    return (wikipedia_search,)


@app.cell
def _(wikipedia_search):
    print(wikipedia_search("Università di Catania"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Implementiamo adesso il nostro `Tool` così che possa essere usato dall'agente.""")
    return


@app.cell
def _(Tool, wikipedia_search):
    wikipedia_tool = Tool(
        name="WikiSearch",
        func=wikipedia_search,
        description="Use this tool to search informations on Wikipedia. Insert the argument (e.g. Einstein, Pythagorean Theorem).",
    )
    return (wikipedia_tool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Com'è possibile notare, il `Tool` presenta una descrizione; essa è **necessaria** affinché l'agente sappia effettivamente cosa farne. Senza di essa, l'agente non saprebbe come o quando utilizzare il `Tool` in questione.""")
    return


@app.cell
def _(ChatOpenAI, MemorySaver, create_react_agent, os, wikipedia_tool):
    tools = [wikipedia_tool]

    memory_tool = MemorySaver()

    llm_openai=ChatOpenAI(
        name="gpt-4.0-mini",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    react_agent_w_tools = create_react_agent(
        model=llm_openai,
        checkpointer=memory_tool,
        tools=tools
    )
    return (react_agent_w_tools,)


@app.cell(hide_code=True)
def _(HumanMessage, react_agent_w_tools):
    def invoke_tool_agent(message: str, debug=False):
        # Estrarre il testo nel modo più sicuro possibile
        text = getattr(message, "content", message)[-1].content

        response = react_agent_w_tools.invoke(
            {"messages": HumanMessage(content=text)},
            config={"configurable": {
                "session_id": "default",
                "thread_id": "default",
                "recursion_limit": 1,
            }},
            debug=debug,
        )
        return response["messages"][-1].content
    return (invoke_tool_agent,)


@app.cell(hide_code=True)
def _(invoke_tool_agent, mo):
    mo.ui.chat(
        invoke_tool_agent,
        prompts=["Ciao!", "Parlami dell'Università di Catania."]
    )
    return


if __name__ == "__main__":
    app.run()
