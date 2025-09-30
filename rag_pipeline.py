from transformers import pipeline
from vector_store import VectorStore  # <-- CORRECTED THIS LINE
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Implements the Retrieval-Augmented Generation logic."""

    def __init__(self, vector_store: VectorStore, qa_model_name='deepset/roberta-base-squad2'):
        self.vector_store = vector_store
        logger.info(f"Loading Question-Answering model: {qa_model_name}")
        self.qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)
        
        self.role_prompts = {
            "Product Lead": "From a business and product performance perspective, focusing on metrics, user behavior, and trends, answer the following question:",
            "Tech Lead": "From a technical standpoint, focusing on system performance, API integrations, errors, and response times, answer the following question:",
            "Compliance Lead": "From a compliance and regulatory viewpoint, focusing on risks, audit trails, and suspicious patterns, answer the following question:",
            "Bank Alliance Lead": "From a partnership and SLA perspective, focusing on integration health and agreement terms with bank partners, answer the following question:",
            "Default": "Based on the provided context, answer the following question:"
        }

    def answer_query(self, query: str, role: str) -> (str, list[str], float):
        """
        Answers a query by retrieving relevant context and using a QA model.
        
        Args:
            query: The user's question.
            role: The stakeholder role of the user.

        Returns:
            A tuple containing the answer, a list of sources, and a confidence score.
        """
        logger.info(f"Performing semantic search for query: '{query}'")
        retrieved_docs = self.vector_store.search(query, k=3)

        if not retrieved_docs:
            return "I couldn't find any relevant information in the uploaded documents to answer your question.", [], 0.0

        # Combine the content of retrieved documents into a single context
        context = " ".join([doc['content'] for doc in retrieved_docs])
        sources = [doc['source'] for doc in retrieved_docs]

        # Get the role-specific instruction
        instruction = self.role_prompts.get(role, self.role_prompts["Default"])
        
        # Prepare the input for the QA model
        qa_input = {
            'question': f"{instruction} {query}",
            'context': context
        }

        logger.info("Generating answer with QA model...")
        result = self.qa_pipeline(qa_input)

        answer = result['answer']
        confidence = result['score']
        
        logger.info(f"Generated answer with confidence: {confidence}")

        return answer, sources, confidence

