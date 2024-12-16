import streamlit as st
from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.knowledge.text import TextKnowledgeBase
from phi.vectordb.qdrant import Qdrant
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from phi.model.anthropic import Claude
from phi.embedder.openai import OpenAIEmbedder
from typing import List, Optional, Dict, Any
import tempfile
import os
import openai
import anthropic
from markitdown import MarkItDown

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
                    
                    # Create text file from converted content
                    text_file_path = os.path.join(temp_dir, "converted.txt")
                    with open(text_file_path, "w") as f:
                        f.write(conversion_result.text_content)
                    
                    # Create and load knowledge base using phidata's TextKnowledgeBase
                    knowledge_base = TextKnowledgeBase(
                        path=temp_dir,
                        vector_db=vector_db,
                        embedder=embedder,
                        recreate_vector_db=True
                    )
                    knowledge_base.load()
                except Exception as e:
                    raise Exception(f"Error processing non-PDF document: {str(e)}")
            
            return knowledge_base
                
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = None
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
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "o1-mini"
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'analysis_response' not in st.session_state:
        st.session_state.analysis_response = None
    if 'key_points_response' not in st.session_state:
        st.session_state.key_points_response = None
    if 'recommendations_response' not in st.session_state:
        st.session_state.recommendations_response = None
    if 'password_entered' not in st.session_state:
        st.session_state.password_entered = False

# Get API keys from environment/secrets
API_KEYS = {
    'openai': os.environ.get('OPENAI_KEY', ''),
    'anthropic': os.environ.get('ANTHROPIC_KEY', ''),
    'qdrant': os.environ.get('QDRANT_KEY', '')
}

def validate_api_key(key: str, key_type: str) -> bool:
    """Validate API key format"""
    if not key:
        return False
    if key_type == 'openai' and not key.startswith('sk-proj-'):
        return False
    if key_type == 'anthropic' and not key.startswith('sk-ant-'):
        return False
    return True

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    
    init_session_state()

    st.title("AI Legal Agent Team 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # Add password input for autofill
        password = st.text_input("Enter password to autofill API keys", type="password")
        if password == "Eromtej12" and not st.session_state.password_entered:
            if all(API_KEYS.values()):
                # Validate keys before setting
                if validate_api_key(API_KEYS['openai'], 'openai') and \
                   validate_api_key(API_KEYS['anthropic'], 'anthropic'):
                    st.session_state.openai_api_key = API_KEYS['openai']
                    st.session_state.anthropic_api_key = API_KEYS['anthropic']
                    st.session_state.qdrant_api_key = API_KEYS['qdrant']
                    st.session_state.password_entered = True
                    st.success("API keys autofilled successfully!")
                else:
                    st.error("Invalid API key format detected. Please check your config.py file.")
            else:
                st.error("Config file not found or API keys not properly configured")
   
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="Enter your OpenAI API key"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key

        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.anthropic_api_key if st.session_state.anthropic_api_key else "",
            help="Enter your Anthropic API key (required for Claude models)"
        )
        if anthropic_key:
            st.session_state.anthropic_api_key = anthropic_key

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
        
        # Add model selection
        st.header("🤖 Model Configuration")
        model_option = st.selectbox(
            "Select Language Model",
            options=["o1-mini", "gpt-4o", "claude-3-5-sonnet-20240620"],
            index=0 if st.session_state.selected_model == "o1-mini" else 1 if st.session_state.selected_model == "gpt-4o" else 2,
            help="Choose the language model to use for analysis"
        )
        if model_option != st.session_state.selected_model:
            st.session_state.selected_model = model_option
            if st.session_state.legal_team:
                st.session_state.legal_team = None
                st.info("Model changed. Please re-upload your document to initialize the team with the new model.")

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
                        
                        # Validate API keys based on selected model
                        if st.session_state.selected_model == "claude-3-5-sonnet-20240620" and not st.session_state.anthropic_api_key:
                            st.error("Anthropic API key is required to use Claude models")
                            return
                        elif st.session_state.selected_model in ["o1-mini", "gpt-4o"] and not st.session_state.openai_api_key:
                            st.error("OpenAI API key is required to use OpenAI models")
                            return

                        # Configure model based on selection
                        if st.session_state.selected_model == "claude-3-5-sonnet-20240620":
                            if not st.session_state.anthropic_api_key:
                                st.error("Anthropic API key is required to use Claude models")
                                return
                            model = Claude(
                                model=st.session_state.selected_model,
                                api_key=st.session_state.anthropic_api_key
                            )
                        else:
                            if not st.session_state.openai_api_key:
                                st.error("OpenAI API key is required to use OpenAI models")
                                return
                            model = OpenAIChat(
                                model=st.session_state.selected_model,
                                api_key=st.session_state.openai_api_key
                            )
                        
                        # Initialize agents
                        legal_researcher = Agent(
                            name="Legal Researcher",
                            role="Legal Research Specialist",
                            model=model,
                            tools=[DuckDuckGo()],
                            knowledge=st.session_state.knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Use DuckDuckGo to search relevant legal databases and external sources for applicable cases and precedents.",
                                "Accurately cite all sources used.",
                                "Provide concise summaries of findings.",
                                "Reference specific sections of the uploaded document when relevant.",
                                "You are giving final reports, do not give partial responses and ask follow up questions.",
                                "Do not create your own sources, use only those provided by DuckDuckGo or the knowledge base."
                            ],
                            show_tool_calls=True,
                            markdown=True
                        )

                        contract_analyst = Agent(
                            name="Contract Analyst",
                            role="Contract Analysis Specialist",
                            model=model,
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Thoroughly review the contract to identify key terms, obligations, and potential issues.",
                                "Reference specific clauses from the uploaded document.",
                                "You are giving final reports, do not give partial responses and ask follow up questions.",
                                "Summarize findings in a clear and concise manner."
                            ],
                            markdown=True
                        )

                        legal_strategist = Agent(
                            name="Legal Strategist", 
                            role="Legal Strategy Specialist",
                            model=model,
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Develop strategic legal recommendations based on the document analysis.",
                                "Provide actionable steps considering identified risks and opportunities.",
                                "You are giving final reports, do not give partial responses and ask follow up questions.",
                                "Ensure recommendations are supported by evidence from the document."
                            ],
                            markdown=True
                        )

                        # Add new Legal Chat Agent
                        legal_chat_agent = Agent(
                            name="Legal Chat Assistant",
                            role="Interactive Legal Consultant",
                            model=model,
                            tools=[DuckDuckGo()],
                            knowledge=knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Engage in interactive discussions about the analyzed document and its legal implications.",
                                "Reference specific parts of the document and previous analysis when answering questions.",
                                "Use DuckDuckGo to search for additional legal information when needed.",
                                "Maintain context of the conversation and previous analyses.",
                                "Provide clear, accurate responses with citations when applicable.",
                                "Ask for clarification if a user's question is ambiguous."
                            ],
                            show_tool_calls=True,
                            markdown=True
                        )

                        # Legal Agent Team
                        st.session_state.legal_team = Agent(
                            name="Legal Team Lead",
                            role="Legal Team Coordinator",
                            model=model,
                            team=[legal_researcher, contract_analyst, legal_strategist, legal_chat_agent],
                            knowledge=st.session_state.knowledge_base,
                            search_knowledge=True,
                            instructions=[
                                "Coordinate efforts between Legal Researcher, Contract Analyst, and Legal Strategist.",
                                "Synthesize findings from all team members to provide comprehensive insights.",
                                "Ensure all recommendations are well-sourced and referenced appropriately.",
                                "Refer to specific sections of the uploaded document as needed.",
                                "Utilize the knowledge base effectively before delegating tasks to team members."
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
                        st.session_state.analysis_response = response
                        
                        # Store the analysis state
                        st.session_state.analysis_complete = True
                        
                        # Display results in tabs
                        st.session_state.analysis_tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                        
                        with st.session_state.analysis_tabs[0]:
                            st.markdown("### Detailed Analysis")
                            if response.content:
                                st.markdown(response.content)
                            else:
                                for message in response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with st.session_state.analysis_tabs[1]:
                            key_points_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:    
                                {response.content}
                                
                                Please summarize the key points in bullet points.
                                Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            st.session_state.key_points_response = key_points_response
                            if key_points_response.content:
                                st.markdown(key_points_response.content)
                            else:
                                for message in key_points_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with st.session_state.analysis_tabs[2]:
                            recommendations_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:
                                {response.content}
                                
                                What are your key recommendations based on the analysis, the best course of action?
                                Provide specific recommendations from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            st.session_state.recommendations_response = recommendations_response
                            if recommendations_response.content:
                                st.markdown(recommendations_response.content)
                            else:
                                for message in recommendations_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

        # Only show chat interface after analysis is complete
        if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
            # Re-display analysis tabs if they exist
            if hasattr(st.session_state, 'analysis_response'):
                tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                
                with tabs[0]:
                    st.markdown("### Detailed Analysis")
                    if st.session_state.analysis_response.content:
                        st.markdown(st.session_state.analysis_response.content)
                    else:
                        for message in st.session_state.analysis_response.messages:
                            if message.role == 'assistant' and message.content:
                                st.markdown(message.content)
                
                with tabs[1]:
                    if hasattr(st.session_state, 'key_points_response'):
                        if st.session_state.key_points_response.content:
                            st.markdown(st.session_state.key_points_response.content)
                        else:
                            for message in st.session_state.key_points_response.messages:
                                if message.role == 'assistant' and message.content:
                                    st.markdown(message.content)
                
                with tabs[2]:
                    if hasattr(st.session_state, 'recommendations_response'):
                        if st.session_state.recommendations_response.content:
                            st.markdown(st.session_state.recommendations_response.content)
                        else:
                            for message in st.session_state.recommendations_response.messages:
                                if message.role == 'assistant' and message.content:
                                    st.markdown(message.content)

            st.divider()
            st.header("💬 Legal Chat Assistant")
            st.info("Ask questions about the document and analysis. The Legal Chat Assistant has access to the document, previous analysis, and can search for additional information.")

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask a question about the document or analysis..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Create context from previous analysis using session state
                        context = f"""
                        Previous document analysis and findings:
                        {st.session_state.analysis_response.content if st.session_state.analysis_response and hasattr(st.session_state.analysis_response, 'content') else ''}
                        
                        Key Points:
                        {st.session_state.key_points_response.content if st.session_state.key_points_response and hasattr(st.session_state.key_points_response, 'content') else ''}
                        
                        Recommendations:
                        {st.session_state.recommendations_response.content if st.session_state.recommendations_response and hasattr(st.session_state.recommendations_response, 'content') else ''}
                        
                        User Question: {prompt}
                        """
                        
                        chat_response = legal_chat_agent.run(context)
                        
                        if chat_response.content:
                            st.markdown(chat_response.content)
                            st.session_state.messages.append({"role": "assistant", "content": chat_response.content})
                        else:
                            for message in chat_response.messages:
                                if message.role == 'assistant' and message.content:
                                    st.markdown(message.content)
                                    st.session_state.messages.append({"role": "assistant", "content": message.content})

            # Add clear chat button
            if st.session_state.messages:
                if st.button("Clear Chat History"):
                    st.session_state.messages = []
                    st.rerun()
    else:
        st.info("Please upload a legal document to begin analysis")

if __name__ == "__main__":
    main() 