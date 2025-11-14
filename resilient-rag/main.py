from src.ingestion import IngestionPipeline

if __name__ == "__main__":

    # Initialize pipeline
    pipeline = IngestionPipeline()

    # Run ingestion process
    chunks = pipeline.run()

    if chunks:
        print("Inspection: First Chunk")
        print(f"Content: {chunks[0].page_content[:200]}")
        print(f"Metadata: {chunks[0].metadata}")