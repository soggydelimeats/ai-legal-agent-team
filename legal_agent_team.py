import streamlit as st
from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.qdrant import Qdrant
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from typing import List, Optional
import tempfile
import os
import openai
from markitdown import MarkItDown

class TextKnowledgeBase:
    """Custom knowledge base class for handling text content"""
    
    def __init__(
        self,
        content: str,
        vector_db: Qdrant,
        embedder: OpenAIEmbedder,
        chunk_size: int = 1500,
        chunk_overlap: int = 100
    ):
        self.content = content
        self.vector_db = vector_db
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            raise ValueError("No text content provided for chunking")
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # If this isn't the first chunk, back up to include overlap
            if start > 0:
                start = start - self.chunk_overlap
            chunk = text[start:end]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            start = end
        return chunks

    def load(self):
        """Process and load the content into the vector database"""
        try:
            # Split content into chunks
            chunks = self._chunk_text(self.content)
            if not chunks:
                raise ValueError("No valid text chunks generated from content")
            
            # Generate embeddings using embed_batch for multiple texts
            embeddings = self.embedder.embed_batch(chunks)
            
            # Store in Qdrant with metadata
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                points.append({
                    "id": i,
                    "vector": embedding,
                    "payload": {
                        "text": chunk,
                        "chunk_index": i
                    }
                })
            
            if not points:
                raise ValueError("No valid embeddings generated")
                
            # Upload to Qdrant
            self.vector_db.upsert(points=points)
            return self
            
        except Exception as e:
            raise Exception(f"Error in TextKnowledgeBase.load(): {str(e)}")

    def search(self, query: str, limit: int = 5) -> List[dict]:
        """Search the knowledge base for relevant content"""
        if not query:
            raise ValueError("No search query provided")
            
        try:
            # Use embed for single text
            query_embedding = self.embedder.embed(query)
            results = self.vector_db.search(
                query_vector=query_embedding,
                limit=limit
            )
            return [result.payload for result in results]
        except Exception as e:
            raise Exception(f"Error in TextKnowledgeBase.search(): {str(e)}")

#initializing the session state variables
def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = None
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = None
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None

def init_qdrant():
    """Initialize Qdrant vector database"""
    if not st.session_state.qdrant_api_key:
        raise ValueError("Qdrant API key not provided")
    if not st.session_state.qdrant_url:
        raise ValueError("Qdrant URL not provided")
        
    return Qdrant(          
        collection="legal_knowledge",
        url=st.session_state.qdrant_url,
        api_key=st.session_state.qdrant_api_key,
        https=True,
        timeout=None,
        distance="cosine"
    )

def process_document(uploaded_file, vector_db: Qdrant):
    """Process document, create embeddings and store in Qdrant vector database"""
    if not st.session_state.openai_api_key:
        raise ValueError("OpenAI API key not provided")
        
    if not uploaded_file:
        raise ValueError("No file uploaded")
        
    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            raise Exception(f"Error saving uploaded file: {str(e)}")

        try:
            # Initialize embedder
            embedder = OpenAIEmbedder(
                model="text-embedding-3-small",
                api_key=st.session_state.openai_api_key
            )

            # Check if file is PDF
            if uploaded_file.name.lower().endswith('.pdf'):
                try:
                    knowledge_base = PDFKnowledgeBase(
                        path=temp_dir, 
                        vector_db=vector_db, 
                        reader=PDFReader(chunk=True),
                        embedder=embedder,
                        recreate_vector_db=True  
                    )
                    knowledge_base.load()
                except Exception as e:
                    raise Exception(f"Error processing PDF: {str(e)}")
            else:
                try:
                    # Initialize MarkItDown for other file types
                    markitdown = MarkItDown(
                        mlm_client=openai.OpenAI(api_key=st.session_state.openai_api_key),
                        mlm_model="gpt-4"
                    )
                    
                    # Convert document to markdown
                    conversion_result = markitdown.convert(temp_file_path)
                    
                    if not conversion_result or not conversion_result.text_content:
                        raise ValueError("Document conversion produced no content")
                    
                    # Create and load knowledge base using the converted content
                    knowledge_base = TextKnowledgeBase(
                        content=conversion_result.text_content,
                        vector_db=vector_db,
                        embedder=embedder
                    )
                    knowledge_base.load()
                except Exception as e:
                    raise Exception(f"Error processing non-PDF document: {str(e)}")
            
            return knowledge_base
                
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 API Configuration")
   
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="Enter your OpenAI API key"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key

        qdrant_key = st.text_input(
            "Qdrant API Key",
            type="password",
            value=st.session_state.qdrant_api_key if st.session_state.qdrant_api_key else "",
            help="Enter your Qdrant API key"
        )
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key

        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url if st.session_state.qdrant_url else "https://2add81f2-7d7c-4617-816c-dbb1b7a85539.europe-west3-0.gcp.cloud.qdrant.io:6333",
            help="Enter your Qdrant instance URL"
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url

        if all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
            try:
                if not st.session_state.vector_db:
                    st.session_state.vector_db = init_qdrant()
                    st.success("Successfully connected to Qdrant!")
            except Exception as e:
                st.error(f"Failed to connect to Qdrant: {str(e)}")

        st.divider()

        if all([st.session_state.openai_api_key, st.session_state.vector_db]):
            st.header("📄 Document Upload")
            uploaded_file = st.file_uploader(
                "Upload Legal Document", 
                type=['pdf', 'docx', 'pptx', 'xlsx', 'txt', 'html', 'htm', 'csv', 'json', 'xml']
            )
            
            if uploaded_file:
                with st.spinner("Processing document..."):
                    try:
                        knowledge_base = process_document(uploaded_file, st.session_state.vector_db)
                        st.session_state.knowledge_base = knowledge_base
                        
                        # Initialize agents
                        legal_researcher = Agent(
                            name="Legal Researcher",
                            role="Legal research specialist",
                            model=OpenAIChat(model="gpt-4o"),
                            tools=[DuckDuckGo()],
                            knowledge=st.session_state.knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Find and cite relevant legal cases and precedents",
                                "Provide detailed research summaries with sources",
                                "Reference specific sections from the uploaded document",
                                "Always search the knowledge base for relevant information"
                            ],
                            show_tool_calls=True,
                            markdown=True
                        )

                        contract_analyst = Agent(
                            name="Contract Analyst",
                            role="Contract analysis specialist",
                            model=OpenAIChat(model="gpt-4o"),
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Review contracts thoroughly",
                                "Identify key terms and potential issues",
                                "Reference specific clauses from the document"
                            ],
                            markdown=True
                        )

                        legal_strategist = Agent(
                            name="Legal Strategist", 
                            role="Legal strategy specialist",
                            model=OpenAIChat(model="gpt-4o"),
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Develop comprehensive legal strategies",
                                "Provide actionable recommendations",
                                "Consider both risks and opportunities"
                            ],
                            markdown=True
                        )

                        # Legal Agent Team
                        st.session_state.legal_team = Agent(
                            name="Legal Team Lead",
                            role="Legal team coordinator",
                            model=OpenAIChat(model="gpt-4o"),
                            team=[legal_researcher, contract_analyst, legal_strategist],
                            knowledge=st.session_state.knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Coordinate analysis between team members",
                                "Provide comprehensive responses",
                                "Ensure all recommendations are properly sourced",
                                "Reference specific parts of the uploaded document",
                                "Always search the knowledge base before delegating tasks"
                            ],
                            show_tool_calls=True,
                            markdown=True
                        )
                        
                        st.success("✅ Document processed and team initialized!")
                            
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")

            st.divider()
            st.header("🔍 Analysis Options")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Contract Review",
                    "Legal Research",
                    "Risk Assessment",
                    "Compliance Check",
                    "Custom Query"
                ]
            )
        else:
            st.warning("Please configure all API credentials to proceed")

    # Main content area
    if not all([st.session_state.openai_api_key, st.session_state.vector_db]):
        st.info("👈 Please configure your API credentials in the sidebar to begin")
    elif not uploaded_file:
        st.info("👈 Please upload a legal document to begin analysis")
    elif st.session_state.legal_team:
        # Create a dictionary for analysis type icons
        analysis_icons = {
            "Contract Review": "📑",
            "Legal Research": "🔍",
            "Risk Assessment": "⚠️",
            "Compliance Check": "✅",
            "Custom Query": "💭"
        }

        # Dynamic header with icon
        st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
  
        analysis_configs = {
            "Contract Review": {
                "query": "Review this contract and identify key terms, obligations, and potential issues.",
                "agents": ["Contract Analyst"],
                "description": "Detailed contract analysis focusing on terms and obligations"
            },
            "Legal Research": {
                "query": "Research relevant cases and precedents related to this document.",
                "agents": ["Legal Researcher"],
                "description": "Research on relevant legal cases and precedents"
            },
            "Risk Assessment": {
                "query": "Analyze potential legal risks and liabilities in this document.",
                "agents": ["Contract Analyst", "Legal Strategist"],
                "description": "Combined risk analysis and strategic assessment"
            },
            "Compliance Check": {
                "query": "Check this document for regulatory compliance issues.",
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Comprehensive compliance analysis"
            },
            "Custom Query": {
                "query": None,
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Custom analysis using all available agents"
            }
        }

        st.info(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 Active Legal AI Agents: {', '.join(analysis_configs[analysis_type]['agents'])}")  #dictionary!!

        # Replace the existing user_query section with this:
        if analysis_type == "Custom Query":
            user_query = st.text_area(
                "Enter your specific query:",
                help="Add any specific questions or points you want to analyze"
            )
        else:
            user_query = None  # Set to None for non-custom queries


        if st.button("Analyze"):
            if analysis_type == "Custom Query" and not user_query:
                st.warning("Please enter a query")
            else:
                with st.spinner("Analyzing document..."):
                    try:
                        # Ensure OpenAI API key is set
                        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                        
                        # Combine predefined and user queries
                        if analysis_type != "Custom Query":
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            Primary Analysis Task: {analysis_configs[analysis_type]['query']}
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            Please search the knowledge base and provide specific references from the document.
                            """
                        else:
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            {user_query}
                            
                            Please search the knowledge base and provide specific references from the document.
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """

                        response = st.session_state.legal_team.run(combined_query)
                        
                        # Display results in tabs
                        tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                        
                        with tabs[0]:
                            st.markdown("### Detailed Analysis")
                            if response.content:
                                st.markdown(response.content)
                            else:
                                for message in response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[1]:
                            st.markdown("### Key Points")
                            key_points_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:    
                                {response.content}
                                
                                Please summarize the key points in bullet points.
                                Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if key_points_response.content:
                                st.markdown(key_points_response.content)
                            else:
                                for message in key_points_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[2]:
                            st.markdown("### Recommendations")
                            recommendations_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:
                                {response.content}
                                
                                What are your key recommendations based on the analysis, the best course of action?
                                Provide specific recommendations from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if recommendations_response.content:
                                st.markdown(recommendations_response.content)
                            else:
                                for message in recommendations_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    else:
        st.info("Please upload a legal document to begin analysis")

if __name__ == "__main__":
    main() 