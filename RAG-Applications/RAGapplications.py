import os
import validators
import streamlit as st

## Langchain Imports
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, UnstructuredURLLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains.summarize import load_summarize_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
#from langchain_chroma.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS

# Langsmith Tracking 
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY") 
os.environ["LANGCHAIN_TRACING_V2"]="true" 
os.environ["LANGCHAIN_PROJECT"]="RAG MEGA PROJECT"


## Set up Streamlit app
st.set_page_config(page_title="AI-Powered Knowledge Hub", page_icon="🤖", layout="wide", menu_items={
                     'Get Help': 'https://docs.streamlit.io/',
                     'Report a bug': 'https://github.com/Chhotoo-11/RAG-Applications/issues',
                     'About': "This app was created by Chhotoo Solanki."
    })
st.title("🤖 AI Knowledge Assistant")


## Sidebar for settings
with st.sidebar:
    st.title("🛠️ Configuration")
    st.write("Configure your app settings below.")
    
    api_mode = st.selectbox("Choose the API Interface",
                            ["Choose an API", "NVIDIA API", "GROQ API"])

    api_key = ""
    if api_mode == "GROQ API":
        api_key = st.text_input("Enter Groq API Key", value="", type="password")
    elif api_mode == "NVIDIA API":
        api_key = st.text_input("Enter NVIDIA API Key", value="", type="password")

    
    st.markdown("---")

    # Section for app navigation
    st.subheader("🌐 App Mode")
    app_mode = st.selectbox("Choose the app mode", 
                            ["Chat with PDF", "URL/YouTube Summarizer", "Web Search"])


# Initialize LLM
def get_llm():
    if api_mode == "GROQ API" and api_key:
         return ChatGroq(groq_api_key=api_key, model_name="Gemma-7b-It")

    elif api_mode == "NVIDIA API" and api_key:
         return ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key=api_key)
    return None

##  Selecting app mode
if api_mode == "Choose an API" or not api_key:
    st.info("Please choose an API and provide your API Key to continue.")
else:
    llm = get_llm()
    if llm:
        st.write(llm)
        if app_mode == "Chat with PDF":
            st.header("📄Chat with PDF")
            st.write("Start interacting with your PDF documents in a chat format. Upload a PDF and ask questions or extract information effortlessly.")

            # Upload a pdf
            uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

            os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
            embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Chat interface
            session_id= st.text_input("Session ID:", value="default_session")

            # Statefully manage chat history
            if "store" not in st.session_state:
                st.session_state.store= {}

            # Process uploaded PDF's:
            if uploaded_files:
                documents= []
                for uploaded_file in uploaded_files:
                    temppdf= f"./temp.pdf"
                    with open(temppdf, 'wb') as file:
                        file.write(uploaded_file.getvalue())
                        file_name= uploaded_file.name
                    
                    loader= PyPDFLoader(temppdf)
                    docs= loader.load()
                    documents.extend(docs)

                # Split and create embeddings for the documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                splits = text_splitter.split_documents(documents)
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                retriever = vectorstore.as_retriever() 

                contextualize_q_system_prompt=(
                    "Given a chat history and the latest user question"
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )

                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                )

                history_aware_retriever= create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

                # Answer question
                system_prompt = (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer "
                        "the question. If you don't know the answer, say that you "
                        "don't know. Use three sentences maximum and keep the "
                        "answer concise."
                        "\n\n"
                        "{context}"
                    )
                qa_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                
                question_answer_chain= create_stuff_documents_chain(llm,qa_prompt)
                rag_chain= create_retrieval_chain(history_aware_retriever,question_answer_chain)

                def get_session_history(session: str)->BaseChatMessageHistory:
                    if session_id not in st.session_state.store:
                        st.session_state.store[session_id]= ChatMessageHistory()
                    return st.session_state.store[session_id]
                
                conversational_rag_chain=RunnableWithMessageHistory(
                    rag_chain,get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )

                user_input = st.text_input("Your question:")
                if user_input:
                    session_history=get_session_history(session_id)
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={
                            "configurable": {"session_id":session_id}
                        },  # constructs a key "abc123" in `store`.
                    )
                    #st.write(st.session_state.store)
                    #st.write("Assistant:", response['answer'])
                    st.success(f"Assistant: {response['answer']}")
                    #st.write("Chat History:", session_history.messages)
            

        ## Web Search
        elif app_mode == "Web Search":
            st.header("Web Search")
            st.write("Easily search the web right from this app. Simply enter your query below to begin.")
    
            ## Tool setup
            arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
            arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

            api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
            wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

            search=DuckDuckGoSearchRun(name="Search")

            if "messages" not in st.session_state:
                st.session_state["messages"]=[
                    {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
                ]

            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg['content'])

            if prompt:=st.chat_input(placeholder="Welcome"):
                st.session_state.messages.append({"role":"user","content":prompt})
                st.chat_message("user").write(prompt)

                tools=[search,arxiv,wiki]

                search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

                with st.chat_message("assistant"):
                    st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
                    response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
                    st.session_state.messages.append({'role':'assistant',"content":response})
                    st.write(response)


        ## 
        elif app_mode == "URL/YouTube Summarizer":
            st.header("URL/YouTube Summarizer")
            st.write("Enter a URL or YouTube link to quickly generate a concise summary of the content")
            
            generic_url=st.text_input("Enter a URL",label_visibility="collapsed")
    
            prompt_template = """
            Provide a summary of the following content in 300 words:
            Content: {text}

            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

            def process_and_summarize(docs):
                # Split the documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    chunk_overlap=200,
                    length_function=len
                )
                texts = text_splitter.split_documents(docs)
                
                # Summarize each chunk
                chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt)
                return chain.run(texts)

            if st.button("Summarize the Content from YT or Website"):
                ## Validate all the inputs
                if not api_key.strip() or not generic_url.strip():
                    st.error("Please provide the information to get started")
                elif not validators.url(generic_url):
                    st.error("Please enter a valid URL. It can be a YouTube video URL or website URL")
                else:
                    try:
                        with st.spinner("Processing..."):
                            ## Loading the website or YT video data
                            if "youtube.com" in generic_url:
                                loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                            else:
                                loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                             )
                            docs=loader.load()
                                        # Process and summarize the content
                            output_summary = process_and_summarize(docs)

                            st.success(output_summary)
                    except Exception as e:
                        st.exception(f"Exception: {e}")

    else:
        st.error("Failed to initialize LLM. Please check your API key and selection.")

