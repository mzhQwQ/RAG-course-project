from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import gradio as gr

# 1. 加载文档
loader = TextLoader("ai_course_notes.txt", encoding="utf-8")
docs = loader.load()

# 2. 切分文本
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 向量模型
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 4. 向量库
db = Chroma.from_documents(chunks, embedding, persist_directory="./chroma_db")
db.persist()
retriever = db.as_retriever(search_kwargs={"k": 3})

# 5. 本地大模型
llm = Ollama(model="qwen2.5:3b", temperature=0.1)

# 6. RAG问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 7. 网页界面
def answer(question):
    return qa_chain.run(question)

demo = gr.Interface(
    fn=answer,
    inputs="text",
    outputs="text",
    title="RAG知识库问答系统",
    description="基于本地文档的AI问答"
)

if __name__ == "__main__":
    demo.launch()