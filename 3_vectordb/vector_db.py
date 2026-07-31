import chromadb
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="new_collection",
)

# Add documents with embeddings
collection.add(
    ids=["id1", "id2", "id3", "id4"],
   documents=[
        "Apple is leading in a smart phone game with iPhone sales up by 35%",
        "Tesla booked a minor profit of 1 billion $ in Q2",
        "Apples are high in fiber, vitamin C, and various antioxidants",
        "SpaceX got NASA contract worth 10 billion $",
    ]
    
)

collection.get(
    include=["documents","embeddings","metadatas"]
)


results = collection.query(
    query_texts=["this is query realted to alon musk"],
    n_results=2
)

print(results)