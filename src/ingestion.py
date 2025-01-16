from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.model import Model
from pathlib import Path
from uuid import uuid4


class Ingestor:
    """
        Class to handle the ingestion of pdf documents into vector store.
    """

    def __init__(self, file_name: str, model: Model, chunk_size=1000, chunk_overlap: int = 300):
        """
        Initialize the Ingestor class and immediately instantiate vector store.

        Args:
            file_name (str): The name of the file to ingest.
            model (Model): The model used for embeddings.
            chunk_size (int): The size of each chunk for splitting documents. Default is 1000.
            chunk_overlap (int): The number of characters to overlap between chunks. Default is 300.
        """
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_folder = Path(__file__).parent.parent / 'data'
        self.persist_directory = Path(__file__).parent.parent / 'db' / 'chroma_langchain_db'
        self.file_path = self.data_folder / file_name

        self._instantiate_vector_store()

    def _instantiate_vector_store(self):
        """
        Instantiate and initialize the vector store using Chroma with the model's embedding.
        This method is called during the initialization of the Ingestor class.
        """
        self.vector_store = Chroma(collection_name="documents",
                                   embedding_function=self.model.embeddings_model,
                                   persist_directory=str(self.persist_directory.absolute()))

    def ingest_file(self):
        """
        Ingest PDF file into the vector store by performing the following steps:
        - Check that the file is a PDF.
        - Load the PDF file.
        - Split the content of the PDF into chunks.
        - Add the chunks to the vector store.

        Raises:
            ValueError: If the file is not a PDF.
        """
        if self.file_path.suffix != '.pdf':
            raise ValueError('The file must be a pdf.')

        # Load the PDF file
        loader = UnstructuredPDFLoader(str(self.file_path.absolute()))
        loaded_documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)
        documents = text_splitter.split_documents(loaded_documents)

        # Generate unique IDs for the documents
        uuids = [str(uuid4()) for _ in range(len(documents))]

        # Add documents to the vector store
        self.vector_store.add_documents(documents=documents, ids=uuids)
