from typing import List, Dict, Optional
import streamlit as st
import json

class QueryTransformer:
    """
    Implements RAG-fusion for query transformation.
    Generates multiple perspectives of the same query to improve retrieval coverage.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.rag_fusion_prompt = """Generate 5 different versions of this query that capture different aspects and perspectives. 
        Each version should help find relevant educational video content.
        {history}
        Current Query: {query}
        
        Consider these perspectives:
        1. Basic explanation/definition
        2. Practical applications/examples
        3. Technical details/methodology
        4. Related concepts/prerequisites
        5. Common challenges/misconceptions
        
        Return ONLY the transformed queries, one per line. Do not include numbers, labels, or any other text."""
        
        self.summary_prompt = """Provide a very concise summary (2-3 sentences) of the key technical concepts and topics discussed in this conversation that are relevant for the next query.

        Conversation:
        {conversation}
        
        Return ONLY the summary, focusing on technical concepts."""
    
    def _summarize_older_messages(self, conversation_pairs: List[tuple]) -> str:
        """Create a concise summary of older messages"""
        if not conversation_pairs or not self.llm:
            return ""
            
        # Format conversation for summarization
        conversation = ""
        for user_msg, asst_msg in conversation_pairs:
            conversation += f"Q: {user_msg}\nA: {asst_msg}\n\n"
            
        messages = [
            {
                "role": "system",
                "content": "You are a technical conversation summarizer. Create very concise summaries focusing on key concepts."
            },
            {
                "role": "user",
                "content": self.summary_prompt.format(conversation=conversation)
            }
        ]
        
        try:
            response = self.llm.invoke(messages)
            summary = response.content if hasattr(response, 'content') else str(response)
            print(f"@@summary strip length: {summary.strip()}")
            st.write(f"🔍 Summarizing older messages: {summary.strip()}")
            return summary.strip()
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return ""
    
    def _get_chat_history(self, max_pairs: int = 2) -> str:
        """Get recent chat history with summarization for longer conversations"""
        if "messages" not in st.session_state:
            return "No previous context."
            
        messages = st.session_state.messages
        pairs = []
        i = 0
        while i < len(messages) - 1:
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                pairs.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 1
        
        if not pairs:
            return "No previous context."
        
        history = ""
        
        # If we have more than max_pairs, summarize older conversations
        if len(pairs) > max_pairs:
            older_pairs = pairs[:-max_pairs]
            summary = self._summarize_older_messages(older_pairs)
            if summary:
                history = "Previous discussion summary:\n" + summary + "\n\n"
            
        # Add the most recent conversations in full
        history += "Recent messages:\n"
        recent_pairs = pairs[-max_pairs:]
        for user_msg, assistant_msg in recent_pairs:
            # Truncate very long messages
            user_msg_short = user_msg[:500] + "..." if len(user_msg) > 500 else user_msg
            asst_msg_short = assistant_msg[:500] + "..." if len(assistant_msg) > 500 else assistant_msg
            history += f"User: {user_msg_short}\nAssistant: {asst_msg_short}\n\n"
        
        return history
    
    def _clean_query(self, query: str) -> str:
        """Clean up query by removing leading numbers, dots, and extra whitespace"""
        # Remove leading numbers and dots (e.g., "1.", "2.", etc.)
        cleaned = query.strip()
        while cleaned and cleaned[0].isdigit():
            cleaned = cleaned[1:].strip()
        if cleaned.startswith('.'):
            cleaned = cleaned[1:].strip()
        return cleaned

    def _generate_diverse_queries(self, query: str) -> List[str]:
        """Generate diverse query variations using the LLM"""
        try:
            if not self.llm:
                return [query]
                
            # Get recent chat history
            history = self._get_chat_history()
                
            # Format prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates diverse search queries for educational video content. Return the queries as a simple list, one per line."
                },
                {
                    "role": "user",
                    "content": self.rag_fusion_prompt.format(history=history, query=query)
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
            
            # Clean and filter queries
            queries = [
                self._clean_query(q) for q in content.split('\n') 
                if q.strip() and not q.startswith('{') and not q.startswith('}') 
                and not q.lower().startswith('here') and not q.lower().startswith('"queries"')
            ]
            
            # Always include original query if not present
            if query not in queries:
                queries.insert(0, query)
            
            return queries
            
        except Exception as e:
            st.error(f"Error generating diverse queries: {str(e)}")
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
            st.write("Generated query variations:\n")
        else:
            st.warning("Could not generate diverse queries, using original query only.")
            
        return queries
