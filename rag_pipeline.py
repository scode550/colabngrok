from transformers import pipeline
from vector_store import VectorStore
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Implements a multi-stage Extract-and-Enhance pipeline."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        # --- Model Loading ---
        # 1. Extractive QA Model (Ground Truth)
        qa_model_name = 'deepset/roberta-base-squad2'
        logger.info(f"Loading Extractive QA model: {qa_model_name}")
        self.qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)

        # 2. Named Entity Recognition Model (Contextual Analysis)
        ner_model_name = 'Jean-Baptiste/roberta-large-ner-english'
        logger.info(f"Loading NER model: {ner_model_name}")
        self.ner_pipeline = pipeline("ner", model=ner_model_name, tokenizer=ner_model_name, grouped_entities=True)

        # 3. Generative Enhancer Model (Fluency and Formatting)
        enhancer_model_name = 'google/flan-t5-base'
        logger.info(f"Loading Generative Enhancer model: {enhancer_model_name}")
        self.enhancer_pipeline = pipeline("text2text-generation", model=enhancer_model_name, tokenizer=enhancer_model_name)

        # --- Role-to-Entity Mapping ---
        # Defines which NER entity types are most relevant for each stakeholder role
        self.ROLE_ENTITY_MAPPING = {
            "Product Lead": ["ORG", "DATE", "MONEY", "PERCENT", "PRODUCT"],
            "Tech Lead": ["ORG", "PRODUCT", "DATE", "CARDINAL", "QUANTITY"],
            "Compliance Lead": ["PERSON", "ORG", "GPE", "LAW", "DATE", "MONEY"],
            "Bank Alliance Lead": ["ORG", "LAW", "DATE", "PERCENT", "GPE"]
        }

    def answer_query(self, query: str, role: str) -> (str, list[str], float):
        """
        Answers a query using the multi-stage Extract-and-Enhance pipeline.
        """
        # === 1. RETRIEVE ===
        logger.info(f"Performing semantic search for query: '{query}'")
        retrieved_docs = self.vector_store.search(query, k=3)
        if not retrieved_docs:
            return "I couldn't find any relevant information in the uploaded documents.", [], 0.0

        context = " ".join([doc['content'] for doc in retrieved_docs])
        sources = list(set([doc['source'] for doc in retrieved_docs])) # Use set for unique sources

        # === 2. EXTRACT (Get Ground Truth with RoBERTa) ===
        logger.info("Extracting raw answer with RoBERTa...")
        qa_input = {'question': query, 'context': context}
        result = self.qa_pipeline(qa_input)
        raw_answer = result['answer']
        confidence = result['score']

        if confidence < 0.2: # Low confidence threshold
            return "I found some related information but could not determine a precise answer.", sources, confidence

        # === 3. ANALYZE (Run NER on the Context) ===
        logger.info("Analyzing context with NER model...")
        ner_results = self.ner_pipeline(context)

        # === 4. FILTER (Select Role-Relevant Entities) ===
        relevant_entity_types = self.ROLE_ENTITY_MAPPING.get(role, [])
        role_specific_entities = [
            f"{entity['word']} ({entity['entity_group']})" 
            for entity in ner_results 
            if entity['entity_group'] in relevant_entity_types
        ]
        # Remove duplicates
        role_specific_entities = list(set(role_specific_entities))
        
        logger.info(f"Found relevant entities for role '{role}': {role_specific_entities}")
        key_entities_str = ", ".join(role_specific_entities) if role_specific_entities else "None"

        # === 5. ENHANCE (Polish the Answer with T5) ===
        logger.info("Enhancing answer with Flan-T5...")
        prompt = f"""
        Rephrase the following "Raw Answer" into a professional, complete sentence.
        Strictly use the information from the "Raw Answer" and the "Key Entities".
        DO NOT add any new information. Your goal is to make the answer fluent and grounded.

        Key Entities: {key_entities_str}
        Raw Answer: {raw_answer}

        Polished Answer:
        """
        
        generated_text = self.enhancer_pipeline(prompt, max_length=256, clean_up_tokenization_spaces=True)
        final_answer = generated_text[0]['generated_text']

        logger.info(f"Final polished answer: {final_answer}")
        
        return final_answer, sources, confidence

