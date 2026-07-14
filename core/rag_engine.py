import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question based only on the meeting transcript context provided below.
If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."
Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain(retriever):
    """Shared chain builder used by both build_rag_chain and load_rag_chain."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def build_rag_chain(transcript: str, collection_name: str = None):
    """Build a fresh vector store from a transcript, then return a ready-to-use RAG chain."""
    vector_store = build_vector_store(transcript, collection_name=collection_name)
    retriever = get_retriever(vector_store)
    return _build_chain(retriever)


def load_rag_chain():
    """Load a previously persisted vector store and return a ready-to-use RAG chain."""
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    return _build_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer


if __name__ == "__main__":
    sample_transcript = (
        "The team discussed the Q3 roadmap and agreed to prioritize the mobile "
        "app redesign. Priya will own the redesign timeline and present it by Friday. "
        "The team also decided to postpone the API migration to Q4."
    )

    chain = build_rag_chain(sample_transcript)
    ask_question(chain, "What did Priya agree to do?")
    ask_question(chain, "What was postponed?")