from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
class QueryTransformer:
    """Handles query transformation using Langchain"""
    
    def __init__(self, llm):
        self.llm = llm
        self.decomposition_chain = None
        self.hyde_chain = None
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup Langchain chains for different transformation methods"""
        if not self.llm:
            return
            
        # Decomposition chain using modern syntax
        decomposition_prompt = PromptTemplate(
            input_variables=["query"],
            template="Decompose the following user query into sub-queries: {query}"
        )
        self.decomposition_chain = decomposition_prompt | self.llm | StrOutputParser()
        
        # HyDE chain using modern syntax
        hyde_prompt = PromptTemplate(
            input_variables=["query"],
            template="Generate a hypothetical document that answers the query: {query}"
        )
        self.hyde_chain = hyde_prompt | self.llm | StrOutputParser()
    
    def transform_query(self, user_query: str, method: str = "decomposition") -> str:
        """Transform query using specified method"""
        if not self.llm:
            st.warning("Langchain LLM not configured. Skipping query transformation.")
            return user_query
        
        if method == "decomposition":
            return self._apply_decomposition(user_query)
        elif method == "hyde":
            return self._apply_hyde(user_query)
        else:
            return user_query
    
    def _apply_decomposition(self, user_query: str) -> str:
        """Apply query decomposition"""
        if not self.decomposition_chain:
            st.warning("Decomposition chain not initialized. Skipping transformation.")
            return user_query
        
        try:
            transformed_query = self.decomposition_chain.invoke({"query": user_query})
            st.info(f"Decomposed Query: {transformed_query}")
            return transformed_query
        except Exception as e:
            st.error(f"Error during query decomposition: {e}")
            return user_query
    
    def _apply_hyde(self, user_query: str) -> str:
        """Apply HyDE transformation"""
        if not self.hyde_chain:
            st.warning("HyDE chain not initialized. Skipping transformation.")
            return user_query
        
        try:
            hypothetical_document = self.hyde_chain.invoke({"query": user_query})
            st.info(f"Hypothetical Document (HyDE): {hypothetical_document[:200]}...")
            return f"Query based on hypothetical document: {hypothetical_document}"
        except Exception as e:
            st.error(f"Error during HyDE generation: {e}")
            return user_query
