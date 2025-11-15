from src.ingestion import IngestionPipeline
from src.vector_store import VectorDBManager

def main():

    print("--- STEP 1: Ingestion ---")

    # Initialize pipeline
    pipeline = IngestionPipeline()

    # Run ingestion process
    chunks = pipeline.run()

    print("\n--- STEP 2: Storage ---")

    if chunks:
        db_manager = VectorDBManager()
        db_manager.upsert_chunks(chunks)

        index_stats = db_manager.pc.Index(
            db_manager.config.INDEX_NAME).describe_index_stats()

        print(f"Total Vectors in DB: {index_stats['total_vector_count']}")

    else:
        print("No chunks generated. Skipping storage")

if __name__ == "__main__":
    main()
