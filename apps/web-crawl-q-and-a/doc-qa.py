import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch, Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter

filedir = "demo/vw.com/"
# Create a list to store the text files


def init_retriever():
    docs = []
    # Get all the text files in the text directory
    for file in os.listdir(filedir):
        loader = TextLoader(filedir + file)

        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200)
    chuncks = text_splitter.split_documents(docs)

    # Create embeddings for chuncks
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # vectordb = DocArrayInMemorySearch.from_documents(chuncks, embeddings)
    vectordb = Milvus.from_documents(
        chuncks,
        embeddings,
        connection_args={"host": "127.0.0.1", "port": "9091"},
    )

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


# App
if __name__ == "__main__":

    # openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    openai_api_key = "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"
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    retriever = init_retriever()

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask me anything!")

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[
                                    retrieval_handler, stream_handler])
            st.session_state.messages.append(
                {"role": "assistant", "content": response})
