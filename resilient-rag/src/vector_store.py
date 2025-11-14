import time
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import Config


class VectorDBManager:

    def __init__(self):
        self.config = Config()
        self._validate_keys()

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

        # Initialize Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config.EMBEDDING_MODEL,
            api_key=self.config.OPENAI_API_KEY
        )

        def _validate_keys(self):
            if not self.config.PINECONE_API_KEY or not self.config.OPENAI_API_KEY:
                raise ValueError(
                    "API keys for Pinecone and OpenAI must be set in environment variables")

        def ensure_index_exists(self):
            """Checks if index exists. If not create one"""
            existing_indexes = [i.name for i in self.pc.list_indexes()]

            if self.config.INDEX_NAME not in existing_indexes:
                print(
                    f"Index '{self.config.INDEX_NAME}' not found. Creating the Pinecone index")

                try:
                    self.pc.create_index(
                        name=self.config.INDEX_NAME,
                        dimension=1536,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud=self.config.CLOUD_PROVIDER,
                            region=self.config.REGION
                        )
                    )

                    # Wait for index to initialize
                    while not self.pc.describe_index(self.config.INDEX_NAME).status['ready']:
                        print("Waiting for index to be ready.")
                        time.sleep(5)

                    print(
                        f"Index '{self.config.INDEX_NAME}' created successfully")

                except Exception as e:
                    print(f"Failed to create index: {e}")
                    raise e

            else:
                print(f"Index '{self.config.INDEX_NAME}' already exists")

        def get_vector_store(self):

            """Returns Pinecone Vector Store instance"""
            return PineconeVectorStore(
                index_name=self.config.INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=self.config.PINECONE_API_KEY
            )
        
        def upsert_chunks(self, chunks):

            """Takes chunks from ingestion and upload them"""
            self.ensure_index_exists()

            print(f"Upserting {len(chunks)} chunks to Pinecone")
            vector_store = self.get_vector_store()

            try:
                vector_store.add_documents(documents=chunks)
                print("Upsert complete!")

            except Exception as e:
                print(f"Failed to upsert chunks: {e}")
                raise e