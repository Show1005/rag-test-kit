import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pathlib import Path
from datetime import datetime
import markdown

# 初期化
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

# ナレッジ読込（JSON + Markdown対応）
def load_knowledge(folder_path: str):
    tag_set = set()
    documents = []
    for filepath in Path(folder_path).glob("**/*"):
        if filepath.suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    tags = item.get("tags", [])
                    tag_set.update(tags)
                    doc = Document(
                        page_content=item["content"],
                        metadata={"title": item.get("title"), "tags": ", ".join(tags)}
                    )
                    documents.append(doc)
        elif filepath.suffix == ".md":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"title": filepath.stem, "tags": "markdown"}
                )
                documents.append(doc)
    return documents, sorted(tag_set)

# 類似ナレッジ検索（スコア付き）
def search_knowledge(db, query, k=3):
    return db.similarity_search_with_score(query, k=k)

# ChatGPTへの質問
def ask_gpt(knowledge, query):
    context = "\n\n".join([doc.page_content for doc, _ in knowledge])
    prompt = f"""
    あなたは探索型テストの専門家です。
    以下のナレッジと質問を参考に、具体的なテスト観点を5つ挙げてください。

    ナレッジ:
    {context}

    質問:
    {query}
    """
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# Markdown保存
def save_report(query, results, answer):
    os.makedirs("output", exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"output/{query.replace(' ', '_')}_{dt}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# 質問: {query}\n\n")
        f.write("## 🔍 類似ナレッジ\n")
        for i, (doc, score) in enumerate(results):
            f.write(f"### {i+1}. {doc.metadata.get('title')} (score: {score:.4f})\n")
            f.write(f"- タグ: {doc.metadata.get('tags')}\n")
            f.write(f"\n{doc.page_content}\n\n")
        f.write("## 🧠 ChatGPTの回答\n\n")
        f.write(answer)
    return fname

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG Test App", layout="wide")
    st.title("🧠 RAG探索型テスト観点ジェネレーター")

    with st.spinner("ナレッジを読み込み中..."):
        docs, all_tags = load_knowledge("knowledge")
        db = Chroma.from_documents(docs, embedding=embedding, collection_name="rag-ui")

    query = st.text_input("質問を入力", value="ログインフォームの異常系テスト")
    selected_tags = st.multiselect("タグで絞り込み（JSONのみ）", options=all_tags)

    if st.button("🔍 検索 & 生成") and query:
        with st.spinner("検索中..."):
            filtered_docs = [doc for doc in docs if all(tag in doc.metadata.get("tags", "") for tag in selected_tags)]
            tmp_db = Chroma.from_documents(filtered_docs, embedding=embedding, collection_name="tmp") if selected_tags else db
            results = search_knowledge(tmp_db, query, k=3)

        st.subheader("📚 類似ナレッジ")
        for i, (doc, score) in enumerate(results):
            st.markdown(f"**{i+1}. {doc.metadata.get('title')}**  ")
            st.markdown(f"スコア: `{score:.4f}`")
            st.markdown(f"タグ: {doc.metadata.get('tags')}")
            st.markdown(doc.page_content)
            st.markdown("---")

        with st.spinner("ChatGPTで回答中..."):
            answer = ask_gpt(results, query)

        st.subheader("🧠 ChatGPTの回答")
        st.markdown(answer)

        if st.button("💾 Markdownとして保存"):
            path = save_report(query, results, answer)
            st.success(f"保存しました: {path}")

if __name__ == "__main__":
    main()
