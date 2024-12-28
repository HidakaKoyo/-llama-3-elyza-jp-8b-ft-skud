import torch
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# NOTE: ==== 1) Markdownファイルの読み込み ====
# NOTE: data/ フォルダにある Markdownファイルを一括で読み込む
# NOTE: DirectoryLoader で拡張子マッチさせることで .md のみを読み込み
data_dir = "data"
loader = DirectoryLoader(data_dir, glob="*.md", loader_cls=TextLoader)
documents = loader.load()

# NOTE: ==== 2) テキスト分割 ====
# NOTE: チャンクサイズやオーバーラップは適宜調整してください
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# NOTE: ==== 3) Embeddingモデルの準備 ====
# NOTE: 日本語対応の埋め込みモデルを指定。下記は一例です
embedding_model_name = (
    "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
)
embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

# NOTE: ==== 4) ベクトルストア(FAISS)を構築 ====
db = FAISS.from_documents(docs, embed_model)

# NOTE: ==== 5) LLMの準備(Llama-3-ELYZA-JP-8B) ====
model_name = "elyza/Llama-3-ELYZA-JP-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # NOTE: GPU環境があれば自動的にfloat16等で動作
    device_map="auto",  # NOTE: GPUメモリ状況に応じて自動配置
)
model.eval()

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1200,
    temperature=0.6,
    top_p=0.9,
)

# NOTE: ==== 6) RetrievalQAチェーンを作成 ====
qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=llm_pipeline),
    chain_type="stuff",  # NOTE: "map_reduce", "refine" なども選択可能
    retriever=db.as_retriever(),
    return_source_documents=True,  # NOTE: 必要に応じてソース文書を返却
)

# NOTE: ==== 7) テストで問い合わせを投げる ====
query = "農業政策の今後について、全体感を教えて下さい。"
result = qa_chain(query)

print("回答:", result["result"])

# NOTE: ソース文書（どのMarkdownチャンクが使われたか）を確認したい場合
# print("参照したチャンク:", result["source_documents"])
