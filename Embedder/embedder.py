from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
from hashlib import sha256
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

def embedd(file):

    # Read the CSVs then grab the sentences inside of it (since we know there is only the first colmun)
    df = pd.read_csv(file)
    sentences = df.iloc[:, 0]

    # Embedd the sentences
    embedding = model.encode(sentences)

    # pair the vector with the text and source file
    embedded = [
        {"id": sha256(text.encode('utf-8')).hexdigest(), "text": text, "vector": embedd, "source": file}
        for text, embedd in zip(sentences, embedding)
    ]

    return embedded

def file_paths(folder_path: Path) -> list:
    # Have a file path list 
    file_paths = []

    # Iterate over the files in the direcorty, if its a file add it to the list as a string path, which we can later send in as a path obj
    for file in folder_path.iterdir():
        if file.is_file():
            file_paths.append(str(file))

    # Return the list
    return file_paths

def search_query(query, client):
    prefixed_text = f"Represent this query for retrieving relevant documents: {query}"

    encoded_query = model.encode([prefixed_text])

    results = client.search(
        collection_name='rolex_info',
        data=encoded_query,
        limit=3,
        output_fields=["text", "source"]
    )
    
    # Return the context block
    context_block = []
    for hit in results[0]:
        
        # Create a block of context for the matches
        block = [f"Distance: {hit['distance']} Source: {hit['entity']['source']}, Text: {hit['entity']['text']} "]
        context_block.append(block)
    
    # Join all the context together with a ------- as the sep
    return '\n---\n'.join(context_block)

def main():
    descriptions_paths = file_paths(Path("Chunker/chunked/Descriptions"))
    waitlists_paths = file_paths(Path("Chunker/chunked/Waitlists"))

    all_paths = descriptions_paths + waitlists_paths

    embeddings = []

    for file in all_paths:
        embedded = embedd(file)
        embeddings.extend(embedded)
    
    # Up;laoded all our data into a VDB
    milvus_client = MilvusClient("db/milvus_local.db")

    # Have Auto ID Disabled since we have our own hashed text as the primary key in order to dedup
    schema = milvus_client.create_schema(
        auto_id= False,
        enable_dynamic_field=True
    )
    
    # insert our own fields
    schema.add_field("id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dimension=1024)

    # Index params for searching
    index_params = milvus_client.prepare_index_params()
    index_params.add_index("vector", metric_type="IP", index_type="IVF_FLAT", params={"nlist": 128})

    # Creating the collection if not already created
    if not milvus_client.has_collection("rolex_info"):
        milvus_client.create_collection(
            collection_name="rolex_info", 
            dimension=1024,
            schema=schema,
            index_params=index_params)
    
    # Insert embedding data
    milvus_client.insert(
        collection_name="rolex_info",
        data=embeddings
    )

    print(f"Successfully inserted {len(embeddings)} items into Milvus.")


if __name__ == "__main__":
    main()
