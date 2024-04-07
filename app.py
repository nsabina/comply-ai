
import modal
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

import vecstore
from utils import pretty_log


image = modal.Image.debian_slim( 
    python_version="3.10"
).pip_install( 
    "langchain",
    "openai",
    "langchain-openai",
    "faiss-gpu",
    "pymongo[srv]==3.12",
    "gradio==3.50.2",
    "langchainhub"
)


stub = modal.Stub(
    name="soc-comply",
    image=image,
    secrets=[
        modal.Secret.from_name("soc-comply-secret")
    ],
    mounts=[
        modal.Mount.from_local_python_packages(
            "vecstore", "docstore", "utils", "prompts"
        )
    ],
)

VECTOR_DIR = vecstore.VECTOR_DIR
#vector_storage = modal.NetworkFileSystem.persisted("vector-vol")
vector_storage = modal.NetworkFileSystem.from_name("vector-vol", create_if_missing=True)


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
)
@modal.web_endpoint(method="GET")
def web(query: str, request_id=None):

    pretty_log(
        f"handling request with client-provided id: {request_id}"
    ) if request_id else None

    answer = qanda.remote(
        query,
        request_id=request_id
    )
    return {"answer": answer}


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    keep_warm=1,
)
def qanda(query: str, request_id=None) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        request_id: A unique identifier for the request.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain_openai import ChatOpenAI

    import vecstore, prompts

    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(
        vecstore.INDEX_NAME, embedding_engine
    )
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    pretty_log(f"running on query: {query}")
    pretty_log("selecting sources by similarity to query")
    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)

    pretty_log("running query against Q&A chain")

    llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.0, max_tokens=300)
    chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=prompts.main,
        document_variable_name="sources",
    )

    #pretty_log(f"input_documents: {sources}, question: {query}")

    result = chain.invoke(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )

    answer = result["output_text"]
    #pretty_log(f"answer: {answer}")
    return answer


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    cpu=8.0,  # use more cpu for vector storage creation
)
def create_vector_index(collection: str = None, db: str = None):
    """Creates a vector index for a collection in the document database."""
    import docstore

    pretty_log("connecting to document store")
    db = docstore.get_database(db)
    pretty_log(f"connected to database {db.name}")

    collection = docstore.get_collection(collection, db)
    pretty_log(f"collecting documents from {collection.name}")
    docs = docstore.get_documents(collection, db)

    pretty_log("splitting into bite-size chunks")
    ids, texts, metadatas = prep_documents_for_vector_storage(docs)

    pretty_log(f"sending to vector index {vecstore.INDEX_NAME}")
    embedding_engine = vecstore.get_embedding_engine(disallowed_special=())
    vector_index = vecstore.create_vector_index(
        vecstore.INDEX_NAME, embedding_engine, texts, metadatas
    )
    vector_index.save_local(folder_path=VECTOR_DIR, index_name=vecstore.INDEX_NAME)
    pretty_log(f"vector index {vecstore.INDEX_NAME} created")


@stub.function(image=image)
def drop_docs(collection: str = None, db: str = None):
    """Drops a collection from the document storage."""
    import docstore

    docstore.drop(collection, db)


def prep_documents_for_vector_storage(documents):
    """Prepare documents from document store for embedding and vector storage.

    Documents are split into chunks so that they can be used with sourced Q&A.

    Arguments:
        documents: A list of LangChain.Documents with text, metadata, and a hash ID.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    ids, texts, metadatas = [], [], []
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas

    return ids, texts, metadatas


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
)
def cli(query: str):
    answer = qanda.remote(query)
    pretty_log("ANSWER")
    print(answer)

web_app = FastAPI(docs_url=None)


@web_app.get("/")
async def root():
    return {"message": "See /gradio for the dev UI."}


@web_app.get("/docs", response_class=RedirectResponse, status_code=308)
async def redirect_docs():
    """Redirects to the Gradio subapi docs."""
    return "/gradio/docs"


@stub.function(
    image=image,
    network_file_systems={
        str(VECTOR_DIR): vector_storage,
    },
    keep_warm=1,
)
@modal.asgi_app(label="soc-comply-backend")
def fastapi_app():
    """A simple Gradio interface for debugging."""
    import gradio as gr
    from gradio.routes import App

    def chain_qanda(*args, **kwargs):
        return qanda.remote(*args, **kwargs)

    inputs = gr.TextArea(
        label="Question",
        value="What types of organizations should comply with SOC2?",
        show_label=True,
    )
    outputs = gr.TextArea(
        label="Answer", value="The answer will appear here.", show_label=True
    )

    interface = gr.Interface(
        fn=chain_qanda,
        inputs=inputs,
        outputs=outputs,
        examples=[
            "What are the key objectives of the GDPR?",
            "How does SOC2 define confidentiality?",
            "What are the requirements for obtaining ISO/IEC 27001 certification?",
            "What is ISO/IEC 27001 about?",
            "How does GDPR address data subject rights?",
            "What types of organizations should comply with SOC2?",
            "What are the main elements of an ISO/IEC 27001 Information Security Management System (ISMS)?",
            "How is personal data defined?",
            "What are the SOC2 trust service criteria?",
            "What is the scope of ISO/IEC 27001's information security standards?",
            "How does SOC2 address confidentiality and privacy?",
            "What are the continuous improvement requirements in ISO/IEC 27001?",
            "What are the implications of GDPR for data processors and controllers?",
            "Каковы основные цели GDPR?"
        ],
        allow_flagging="never",
        theme=gr.themes.Glass(radius_size="none", text_size="lg"),
        css=".gradio-container {background-color: rgba(229, 240, 244, 0.8); font-family: 'Roboto', sans-serif; color: #91a1a8}"
    )

    interface.dev_mode = False
    interface.config = interface.get_config_file()
    interface.validate_queue_settings()
    #interface.allowed_paths = [absolute_path]
    gradio_app = App.create_app(
        interface, app_kwargs={"docs_url": "/docs", "title": "soc-comply-app"}
    )

    @web_app.on_event("startup")
    async def start_queue():
        if gradio_app.get_blocks().enable_queue:
            gradio_app.get_blocks().startup_events()

    web_app.mount("/gradio", gradio_app)

    return web_app
