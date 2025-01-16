from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from src.ingestion import Ingestor
from src.model import Model
from typing import List, Tuple, Optional


class PdfChat:
    """
    Class to handle the interactive chat
    """

    def __init__(self, model: Model, ingestor: Ingestor, prompt_message: Optional[List[Tuple[str]]] = None):
        """
        Initialize the PdfChat instance with the model, ingestor, and an optional custom prompt message.

        Immediately defines the chat prompt and sets up the retrieval chain for
        generating responses based on the ingested PDF data.

        Args:
            model (Model): The model used for generating responses.
            ingestor (Ingestor): The Ingestor object responsible for managing the vector store.
            prompt_message (Optional[List[Tuple[str]]]): A custom prompt message to guide the assistant's responses.
        """

        self.model = model
        self.ingestor = ingestor
        self.prompt_message = prompt_message
        self._define_prompt()
        self._define_retrieval_chain()

    def _define_prompt(self):
        """
        Defines the system and user prompt messages to guide the assistant's behavior during the chat
        """

        if not self.prompt_message:
            prompt_message = [
                ('system', 'You are an excellent and helpful assistant. '
                           'Answer the question based only on the data provided.'),
                ('human', 'Use the user question {input} to answer the question. Use only the {context} '
                          'to answer the question.')
            ]

            self.prompt_message = prompt_message

        self.prompt = ChatPromptTemplate.from_messages(self.prompt_message)

    def _define_retrieval_chain(self):
        """
        Sets up the retrieval chain that retrieves context from the vector store and combines it
        with the prompt template to generate the final response.
        """

        retriever = self.ingestor.vector_store.as_retriever(kwargs={"k": 10})

        combine_docs_chain = create_stuff_documents_chain(self.model.chat_model, self.prompt)

        self.retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    def chat(self):
        """
        Starts the chat interaction, allowing the user to ask questions based on the ingested PDF data.
        """

        while True:
            query = input("Start the chat! \nTo quit, type 'q': ")
            if query.lower() == 'q':
                break

            result = self.retrieval_chain.invoke({"input": query})
            print("Assistant: ", result["answer"], "\n\n")


if __name__ == '__main__':
    model_instance = Model(embeddings_model='nomic-embed-text:latest',
                           chat_model='llama3.2:1b')

    ingestion = Ingestor(file_name='testpdf.pdf',
                         model=model_instance)
    ingestion.ingest_file()

    chat = PdfChat(model=model_instance, ingestor=ingestion)
    chat.chat()
