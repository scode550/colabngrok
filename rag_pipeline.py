from transformers import pipeline, Pipeline
from vector_store import VectorStore
import logging

# Set up a specific logger for this module
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Implements a robust, state-less, multi-stage RAG pipeline.
    The models are loaded on initialization, and the pipeline can then be
    used for any number of queries without reloading.
    """

    def __init__(self):
        """
        Initializes the pipeline and loads all required models.
        This is a heavy operation and should only be done once.
        """
        logger.info("Initializing RAG Pipeline and loading models...")
        try:
            self.qa_pipeline: Pipeline = pipeline("question-answering", model='deepset/roberta-base-squad2')
            self.ner_pipeline: Pipeline = pipeline("ner", model='Jean-Baptiste/roberta-large-ner-english', grouped_entities=True)
            self.enhancer_pipeline: Pipeline = pipeline("text2text-generation", model='google/flan-t5-base')
            logger.info("All models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load one or more models: {e}")
            raise

        # Mapping of roles to the NER entity types they care about.
        self.ROLE_ENTITY_MAPPING = {
            "Product Lead": ["ORG", "DATE", "MONEY", "PERCENT", "PRODUCT"],
            "Tech Lead": ["ORG", "PRODUCT", "DATE", "CARDINAL", "QUANTITY"],
            "Compliance Lead": ["PERSON", "ORG", "GPE", "LAW", "DATE", "MONEY"],
            "Bank Alliance Lead": ["ORG", "LAW", "DATE", "PERCENT", "GPE"]
        }
        
        # Keywords to detect if the user is requesting a synthetic task (e.g., summarization).
        self.TASK_KEYWORDS = ["list", "summarize", "extract", "show me all", "what are all", "find all"]

    def _is_task_oriented(self, query: str) -> bool:
        """Checks if a query is task-oriented based on keywords."""
        return any(keyword in query.lower() for keyword in self.TASK_KEYWORDS)

    def answer_query(self, query: str, role: str, vector_store: VectorStore) -> (str, list[str], float):
        """
        Answers a query using a multi-stage process. This method is state-less
        and depends on the provided vector_store for context.
        """
        logger.info(f"Processing query: '{query}' for role: '{role}'")
        
        # --- 1. RETRIEVE relevant document chunks ---
        k_docs = 5 if self._is_task_oriented(query) else 3
        retrieved_docs = vector_store.search(query, k=k_docs)
        if not retrieved_docs:
            logger.warning("No relevant documents found in vector store for the query.")
            return "I couldn't find any relevant information in the uploaded documents to answer your question.", [], 0.0

        context = " ".join([doc['content'] for doc in retrieved_docs])
        sources = sorted(list(set([doc['source'] for doc in retrieved_docs])))

        try:
            # --- 2. DETECT query type and EXECUTE appropriate pipeline ---
            if self._is_task_oriented(query):
                logger.info("Query detected as TASK-ORIENTED. Using generative task prompt.")
                prompt = f"""
                Based ONLY on the following context, perform the requested task. Be concise.
                If the context is insufficient to perform the task, state that clearly.

                Context:
                "{context}"

                Task: "{query}"

                Result:
                """
                task_result = self.enhancer_pipeline(prompt, max_length=512, clean_up_tokenization_spaces=True)
                final_answer = task_result[0]['generated_text']
                confidence = 0.95 # Confidence is high as it's a generative task
                
            else:
                logger.info("Query detected as Q&A. Using Extract-and-Enhance pipeline.")
                
                # --- Stage 1: EXTRACT a direct answer ---
                qa_input = {'question': query, 'context': context}
                result = self.qa_pipeline(qa_input)
                raw_answer = result['answer']
                confidence = result['score']

                if confidence < 0.15: # Increased threshold slightly
                    logger.warning(f"Low confidence ({confidence:.2f}) for query. Returning generic response.")
                    return "I found some related information, but I am not confident enough to provide a precise answer. You may want to rephrase your question.", sources, confidence

                # --- Stage 2: ANALYZE for key entities ---
                ner_results = self.ner_pipeline(context)
                relevant_entity_types = self.ROLE_ENTITY_MAPPING.get(role, [])
                role_specific_entities = list(set([
                    f"{entity['word']} ({entity['entity_group']})" 
                    for entity in ner_results 
                    if entity['entity_group'] in relevant_entity_types
                ]))
                key_entities_str = ", ".join(role_specific_entities) if role_specific_entities else "None provided"

                # --- Stage 3: ENHANCE the answer with context ---
                prompt = f"""
                Rephrase the following "Raw Answer" into a professional, standalone sentence.
                Use ONLY the information from the "Raw Answer". You may reference the "Key Entities" for context.
                Do not add any new information. Your response must be concise.

                Key Entities: {key_entities_str}
                Raw Answer: {raw_answer}

                Polished Answer:
                """
                generated_text = self.enhancer_pipeline(prompt, max_length=256, clean_up_tokenization_spaces=True)
                final_answer = generated_text[0]['generated_text']

        except Exception as e:
            logger.error(f"An error occurred during model inference: {e}")
            return "I encountered an error while processing your request. Please try again.", [], 0.0

        logger.info(f"Final answer generated: {final_answer}")
        return final_answer, sources, confidence

