from src.ingestion import Ingestor
from src.model import Model
from src.pdf_chat import PdfChat

if __name__ == '__main__':
    # Step 1: Create an instance of the Model, specifying the embeddings model and chat model.
    model_instance = Model(embeddings_model='nomic-embed-text:latest',
                           chat_model='llama3.2:1b')

    # Step 2: Initialize the Ingestor with the PDF file and the model instance.
    ingestion = Ingestor(file_name='testpdf.pdf', model=model_instance)

    # Step 3: Ingest the PDF file, which involves loading and splitting the file into chunks,
    # and adding those chunks to the vector store.
    ingestion.ingest_file()

    # Step 4: Create an instance of the PdfChat, passing the model instance and the ingested data.
    chat = PdfChat(model=model_instance, ingestor=ingestion)

    # Step 5: Start the chat loop, where the user can interact with the assistant and get answers.
    chat.chat()
