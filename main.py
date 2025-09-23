import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader

# 1. Set your API key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# 2. Load your PDF
pdf_path = "sample.pdf"   # <-- replace with your file
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 4. Create embeddings & vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Build Retrieval QA chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# 6. Ask questions
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa.run(query)
    print("Answer:", answer)
