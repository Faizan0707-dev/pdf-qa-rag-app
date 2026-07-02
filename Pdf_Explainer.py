import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

parser = StrOutputParser()
prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, say:
"I couldn't find the answer in the document."
Context:
{text}
Question:
{query}
Answer:
""",
    input_variables=["text", "query"]
)

load_dotenv()
model = ChatGroq(model = "openai/gpt-oss-120b")
st.header("Pdf explainer 📝")
st.subheader("Enter any text pdf for explaination and Q&A")
@st.cache_resource
def build_vectorstore(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding)
        return vectorstore
if "QUE_ANS_list" not in st.session_state:
    st.session_state.QUE_ANS_list = []
uploaded_file = st.file_uploader("select yout text pdf file")
if uploaded_file:
    with open ("file.pdf","wb") as f:
        f.write(uploaded_file.read())
        vectorstore = build_vectorstore("file.pdf")
    retriever__ = vectorstore.as_retriever(search_kwargs = {"k":4})
    query = st.text_input("ask any Que. whose answer you want from your your document")
    button = st.button("Enter->")
    if query :
        relevant_doc = retriever__.invoke(query)
        text = "  ".join(doc.page_content for doc in relevant_doc)
        chain = prompt | model | parser
        if button:
            st.success("Generating answer, please wait...")
            final_ans = chain.invoke({"query" : query , "text" : text})
            st.write(final_ans)
            st.session_state.QUE_ANS_list.append((query, final_ans))
if st.button("Delete all chat history 🗑️"):
     st.session_state.QUE_ANS_list = []
     st.rerun()
st.write("Chat history")
for q, a in st.session_state.QUE_ANS_list:
    st.write(f"**Que:** {q}")
    st.write(f"**Ans:** {a}")
    st.write("---"*10)
    st.write("---"*10)
    st.write(" "*10)
st.write(" "*10)
st.write("HOST --->  Faizan Mirza")
