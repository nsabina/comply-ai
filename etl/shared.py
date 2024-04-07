import modal

# definition of our container image and app for deployment on Modal
# see app.py for more details
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "langchain", 
    "pymongo"
)

stub = modal.Stub(
    name="etl-compliance-shared",
    secrets=[
        modal.Secret.from_name("soc-comply-secret"),
    ],
    mounts=[
        # we make our local modules available to the container
        modal.Mount.from_local_python_packages("docstore", "utils")
    ],
)


@stub.function(image=image)
def add_to_document_db(documents_json, collection="dos_content", db="soc-comply"):
    """Adds a collection of json documents to a database."""
    from pymongo import InsertOne

    import docstore

    collection = docstore.get_collection(collection, db)

    requesting, CHUNK_SIZE = [], 250

    for document in documents_json:
        requesting.append(InsertOne(document))

        if len(requesting) >= CHUNK_SIZE:
            collection.bulk_write(requesting)
            requesting = []

    if requesting:
        collection.bulk_write(requesting)


def enrich_metadata(pages):
    """Add metadata: sha256 hash and ignore flag."""
    import hashlib

    for page in pages:
        m = hashlib.sha256()
        m.update(page["text"].encode("utf-8", "replace"))
        page["metadata"]["sha256"] = m.hexdigest()
        if page["metadata"].get("is_endmatter"):
            page["metadata"]["ignore"] = True
        else:
            page["metadata"]["ignore"] = False
    return pages


def chunk_into(list, n_chunks):
    for ii in range(0, n_chunks):
        yield list[ii::n_chunks]


def unchunk(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

