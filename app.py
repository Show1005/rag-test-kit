import os
import json
import streamlit as st
from rag_utils.openai_client import get_openai_client
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pathlib import Path
from datetime import datetime
import markdown
from collections import defaultdict

# 初期化
client = get_openai_client()
embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

PERSIST_DIR = "chroma_db"
KNOWLEDGE_DIR = "knowledge"
OUTPUT_DIR = "output"
HISTORY_FILE = "history.json"

# ナレッジ読込（JSON + Markdown対応）
def load_knowledge(folder_path: str):
    tag_set = set()
    documents = []
    usage_counter = defaultdict(int)
    file_map = defaultdict(list)
    for filepath in Path(folder_path).glob("**/*"):
        if filepath.suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    tags = item.get("tags", [])
                    tag_set.update(tags)
                    doc = Document(
                        page_content=item["content"],
                        metadata={"title": item.get("title"), "tags": ", ".join(tags), "file": filepath.name, "index": i}
                    )
                    documents.append(doc)
                    file_map[filepath.name].append(item)
        elif filepath.suffix == ".md":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"title": filepath.stem, "tags": "markdown"}
                )
                documents.append(doc)
    return documents, sorted(tag_set), file_map

# 類似ナレッジ検索（スコア付き）
def search_knowledge(db, query, k=3):
    return db.similarity_search_with_score(query, k=k)

# ChatGPTへの質問
def ask_gpt(knowledge, query):
    context = "\n\n".join([doc.page_content for doc, _ in knowledge])
    prompt = f"""
    あなたは探索型テストの専門家です。
    以下のナレッジと質問を参考に、具体的なテスト観点を5つ、番号付き箇条書きで挙げてください。

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

# GPTでナレッジ本文を自動生成
def generate_knowledge_body(title, tags):
    prompt = f"""
    タイトル: {title}
    タグ: {', '.join(tags)}
    この情報をもとに、探索的テストで役立つナレッジ文章を作成してください。
    """
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ナレッジ保存
def save_knowledge_json(title, tags, content, filename=None, index=None):
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    if not filename:
        filename = f"{datetime.now().strftime('%Y%m%d')}_custom.json"
    path = os.path.join(KNOWLEDGE_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    if index is not None and 0 <= index < len(data):
        data[index] = {"title": title, "tags": tags, "content": content}
    else:
        data.append({"title": title, "tags": tags, "content": content})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

# Markdown保存
def save_report(query, results, answer):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{OUTPUT_DIR}/{query.replace(' ', '_')}_{dt}.md"
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

# 検索履歴保存
def save_history(query):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append({"query": query, "timestamp": datetime.now().isoformat()})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-10:], f, ensure_ascii=False, indent=2)

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG Test App", layout="wide")
    st.title("🧠 RAG探索型テスト観点ジェネレーター")

    with st.spinner("ナレッジを読み込み中..."):
        docs, all_tags, file_map = load_knowledge(KNOWLEDGE_DIR)
        db = Chroma.from_documents(docs, embedding=embedding, collection_name="rag-ui", persist_directory=PERSIST_DIR)
        db.persist()

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 検索と生成", "📝 ナレッジ登録", "📊 状態・履歴", "✏️ 編集・削除"])

    with tab1:
        query = st.text_input("質問を入力", value="ログインフォームの異常系テスト")
        selected_tags = st.multiselect("タグで絞り込み（JSONのみ）", options=all_tags)

        if st.button("🔍 検索 & 生成") and query:
            with st.spinner("検索中..."):
                filtered_docs = [doc for doc in docs if all(tag in doc.metadata.get("tags", "") for tag in selected_tags)]
                tmp_db = Chroma.from_documents(filtered_docs, embedding=embedding, collection_name="tmp", persist_directory=PERSIST_DIR) if selected_tags else db
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

            if st.button("💾 この観点をナレッジとして保存"):
                save_knowledge_json(title=f"観点：{query}", tags=selected_tags, content=answer)
                st.success("ナレッジとして保存しました")

            if st.button("📄 Markdownとして保存"):
                path = save_report(query, results, answer)
                st.success(f"保存しました: {path}")

            save_history(query)

    with tab2:
        st.markdown("### ナレッジを手動で登録")
        title = st.text_input("タイトル")
        tags_input = st.text_input("タグ（カンマ区切り）")
        gen_body = st.checkbox("GPTに本文を書かせる")
        content = st.text_area("本文", height=200)

        if st.button("➕ ナレッジを登録"):
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
            if gen_body:
                with st.spinner("GPTがナレッジ本文を生成中..."):
                    content = generate_knowledge_body(title, tags)
            save_knowledge_json(title=title, tags=tags, content=content)
            st.success("ナレッジを保存しました")

    with tab3:
        st.write("📦 ベクトルDBの永続化パス:", PERSIST_DIR)
        st.write(f"🧾 ナレッジ数: {len(docs)} 件")
        st.write("📁 タグ一覧:")
        st.write(", ".join(all_tags))
        if os.path.exists(HISTORY_FILE):
            st.markdown("### 🔁 最近の検索履歴")
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                for item in json.load(f)[::-1]:
                    st.markdown(f"- {item['query']} ({item['timestamp'][:19]})")

    with tab4:
        st.markdown("### ✏️ ナレッジの編集・削除")
        selected_file = st.selectbox("ファイルを選択", list(file_map.keys()))
        if selected_file:
            items = file_map[selected_file]
            titles = [item["title"] for item in items]
            selected_index = st.selectbox("ナレッジを選択", list(enumerate(titles)), format_func=lambda x: x[1])
            if selected_index:
                idx = selected_index[0]
                item = items[idx]
                new_title = st.text_input("タイトル", value=item["title"], key="edit_title")
                new_tags = st.text_input("タグ（カンマ区切り）", value=", ".join(item.get("tags", [])), key="edit_tags")
                new_content = st.text_area("本文", value=item["content"], height=200, key="edit_content")

                if st.button("💾 編集内容を保存"):
                    save_knowledge_json(new_title, [t.strip() for t in new_tags.split(",")], new_content, filename=selected_file, index=idx)
                    st.success("ナレッジを更新しました")

                if st.button("🗑️ このナレッジを削除"):
                    del items[idx]
                    save_knowledge_json("", [], "", filename=selected_file, index=None)  # 保存のトリガー用
                    with open(os.path.join(KNOWLEDGE_DIR, selected_file), "w", encoding="utf-8") as f:
                        json.dump(items, f, ensure_ascii=False, indent=2)
                    st.success("ナレッジを削除しました")

if __name__ == "__main__":
    main()
