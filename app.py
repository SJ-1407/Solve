from langchain_community.document_loaders import PyMuPDFLoader

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import YoutubeLoader
import os
from langchain import hub
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from utils import text_to_speech, autoplay_audio, speech_to_text

from streamlit_float import *
from streamlit_mic_recorder import mic_recorder

#from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI
persist_directory = "chroma_db"

api_key = None
if api_key  is None:
    ## Input the Groq API Key
      os.environ['OPENAI_API_KEY']=st.text_input("Enter your open ai  API key:",type="password")
      api_key=os.getenv("OPENAI_API_KEY")


if api_key:
    def vectorstore_exists(directory):
        return os.path.exists(directory) and os.path.isdir(directory)

    if vectorstore_exists(persist_directory):
        # Load the existing vectorstore
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(api_key=api_key),
            persist_directory=persist_directory
        )
        
    else:

        loader=PyMuPDFLoader("Apple_Vision_Pro_Privacy_Overview.pdf")
        data_pdf = loader.load()

        urls = [
            "https://www.apple.com/apple-vision-pro/",
            "https://www.apple.com/apple-vision-pro/specs/",
            "https://www.wired.com/review/apple-vision-pro/",
            "https://www.theverge.com/24054862/apple-vision-pro-review-vr-ar-headset-features-price",
            "https://www.cnet.com/tech/computing/apple-vision-pro-one-month-later-its-in-my-life-sometimes/",
            "https://support.apple.com/en-in/guide/apple-vision-pro/tan39b6bab8f/visionos",
            "https://support.apple.com/en-in/guide/apple-vision-pro/tan489cfe222/visionos",
            "https://support.apple.com/en-in/guide/apple-vision-pro/tan1e660fd7d/visionos",
            "https://support.apple.com/en-in/guide/apple-vision-pro/tan2f5f77158/visionos",
            "https://support.apple.com/en-in/guide/apple-vision-pro/tan836d673da/visionos",
            "https://support.apple.com/en-in/guide/apple-vision-pro/dev1d800e084/visionos",
            "https://support.apple.com/en-in/guide/apple-vision-pro/tan103f047f4/visionos",
        ]



        loader_1 = UnstructuredURLLoader(urls=urls)

        data_web = loader.load()

        data_wiki = WikipediaLoader(query="Apple Vision Pro", load_max_docs=1).load()

        data_yt=[]
        urls_yt=["https://www.youtube.com/watch?v=TX9qSaGXFyg",
                "https://www.youtube.com/watch?v=Vb0dG-2huJE",
                "https://www.youtube.com/watch?v=a_DvwB3aO5U",
                "https://www.youtube.com/watch?v=SaneSRqePVY"
                ]

        for i in urls_yt: 
            loader_yt = YoutubeLoader.from_youtube_url(
            i,
                add_video_info=False
        )
            data_yt.extend(loader_yt.load())


        all_data = data_pdf + data_web + data_yt +data_wiki



        # Load environment variables from the .env file
        load_dotenv()

        # Access the OpenAI API key


        plit_docs = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        split_docs=[]
        for doc in all_data:
            split_docs.extend(splitter.split_documents([doc]))

        # Create and store documents in Chroma database
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = Chroma(embedding_function=embeddings,persist_directory="chroma_db")

        # Add split documents to Chroma database
        vectorstore.add_documents(split_docs)

    llm = OpenAI(openai_api_key=api_key)
    retriever = vectorstore.as_retriever()


    system_prompt = (
        "You are a sales assistant for question-answering tasks.Your main aim is to provide the best possible answer to the user's question regardig Apple vision pro. You should provide the answer in a way to lure the customer to buy the product. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    float_init()

    def initialize_session_state():
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! What question do you have reagrding Apple vision pro?"}
            ]


    initialize_session_state()

    st.title("Apple Vision Pro Help")
    footer_container = st.container()

    with footer_container:
     cols = st.columns([0.8,3],gap="small")  


    with cols[0]:  
        audio_bytes = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=True,
            use_container_width=False,
            callback=None,
            args=(),
            kwargs={},
            key=None
        )
        if(audio_bytes):
         audio_bytes=audio_bytes['bytes']

    with cols[1]: 
    
        if("input" in st.session_state):
            st.text_input("Enter your text query",key="input")
            user_text_input = st.session_state.input
        else:
            user_text_input=st.text_input("Enter your text query",key="input")
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    
    if audio_bytes:
        # Write the audio bytes to a file
        with st.spinner("Transcribing..."):
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            transcript = speech_to_text(webm_file_path)
            if transcript:
                st.session_state.messages.append({"role": "user", "content": transcript})
                with st.chat_message("user"):
                    st.write(transcript)
                os.remove(webm_file_path)

    elif (user_text_input):


        st.session_state.messages.append({"role": "user", "content": user_text_input})
        with st.chat_message("user"):
            st.write(user_text_input)


        
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking🤔..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                final_response = retrieval_chain.invoke({'input':st.session_state.messages[-1]["content"]})["answer"]
                if(final_response.startswith("System:")):
                 final_response = final_response.replace("System:", "")
                if (final_response.startswith("SalesAssistant:")):
                 final_response = final_response.replace("SalesAssistant:","")
        
                print("answer:",final_response)
            with st.spinner("Generating audio response..."):    
                audio_file = text_to_speech(final_response)
                autoplay_audio(audio_file)
            st.write(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            os.remove(audio_file)
    
    # Float the footer container and provide CSS to target it with
    footer_container.float("bottom: 0rem;")
