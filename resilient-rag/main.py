from src.ingestion import IngestionPipeline
from src.vector_store import VectorDBManager
from src.rag import RAGEngine


def main():

    # print("--- STEP 1: Ingestion ---")

    # # Initialize pipeline
    # pipeline = IngestionPipeline()

    # # Run ingestion process
    # chunks = pipeline.run()

    # print("\n--- STEP 2: Storage ---")

    # if chunks:
    #     db_manager = VectorDBManager()
    #     db_manager.upsert_chunks(chunks)

    #     index_stats = db_manager.pc.Index(
    #         db_manager.config.INDEX_NAME).describe_index_stats()

    #     print(f"Total Vectors in DB: {index_stats['total_vector_count']}")

    # else:
    #     print("No chunks generated. Skipping storage")

    print("\n--- STEP 3: RAG Query ---")
    rag = RAGEngine()

    # Interactive query loop
    print("Type 'exit' to quit")

    while True:
        user_query = input("\nAsk a question about GDPR: ")

        if user_query.lower() == "exit":
            break

        result = rag.query(user_query)
        print(f"\nAnswer:\n{result['answer']}")

        print("\nSources Documents used:")
        for i, meta in enumerate(result['source_metadata']):
            print(f"   {i+1}. Source: {meta.get('source', 'Unknown')}")


if __name__ == "__main__":
    main()
