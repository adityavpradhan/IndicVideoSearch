from typing import List, Dict, Optional
import streamlit as st

class QueryTransformer:
    """
    Implements RAG-fusion for query transformation.
    Generates multiple perspectives of the same query to improve retrieval coverage.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.rag_fusion_prompt = """Generate 5 different versions of this query that capture different aspects and perspectives. 
        Each version should help find relevant educational video content.
        
        Original Query: {query}
        
        Consider these perspectives:
        1. Basic explanation/definition
        2. Practical applications/examples
        3. Technical details/methodology
        4. Related concepts/prerequisites
        5. Common challenges/misconceptions
        
        Return only the transformed queries, one per line."""
    
    def _generate_diverse_queries(self, query: str) -> List[str]:
        """Generate diverse query variations using the LLM"""
        try:
            if not self.llm:
                return [query]
                
            # Format prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates diverse search queries for educational video content."
                },
                {
                    "role": "user",
                    "content": self.rag_fusion_prompt.format(query=query)
                }
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if isinstance(response, str):
                content = response
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
                
            # Split into individual queries and clean up
            queries = [q.strip() for q in content.split('\n') if q.strip()]
            
            # Always include original query
            if query not in queries:
                queries.insert(0, query)
                
            return queries
            
        except Exception as e:
            st.error(f"Error generating diverse queries: {e}")
            return [query]
    
    def transform_query(self, user_query: str, method: str = "rag_fusion") -> List[str]:
        """
        Transform the user query using RAG-fusion approach.
        Returns a list of diverse query variations to be used for retrieval.
        """
        if method != "rag_fusion":
            return [user_query]
            
        st.write("🔄 Applying RAG-fusion to generate diverse queries...")
        
        # Generate diverse query variations
        queries = self._generate_diverse_queries(user_query)
        
        # Display generated queries
        if len(queries) > 1:
            st.info("Generated query variations:")
            for i, q in enumerate(queries):
                st.write(f"{i+1}. {q}")
        else:
            st.warning("Could not generate diverse queries, using original query only.")
            
        return queries
