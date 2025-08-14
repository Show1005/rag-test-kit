import os
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pathlib import Path
import markdown

# 環境変数読み込み
load_dotenv()

# OpenAIクライアント初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embeddingモデル初期化（節約版）
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

# ナレッジ読み込み関数（JSON & Markdown対応）
def load_knowledge(folder_path: str, tag_filter: list[str] = None):
    documents = []
    for filepath in Path(folder_path).glob("**/*"):
        if filepath.suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    tags = item.get("tags", [])
                    if tag_filter and not any(tag in tags for tag in tag_filter):
                        continue
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
    return documents

# ベクトルDB作成（永続化せず都度作成）
def create_vectorstore(docs):
    return Chroma.from_documents(docs, embedding=embedding, collection_name="filtered-knowhow")

# 類似ナレッジを検索（スコア付き）
def search_knowledge(db, query, k=3):
    return db.similarity_search_with_score(query, k=k)

# 回答生成
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
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# レポート出力
def save_report_md_html(query, results, answer, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    base = Path(outdir) / query.replace(" ", "_")

    # Markdown保存
    md_path = base.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 質問: {query}\n\n")
        f.write("## 🔍 類似ナレッジ\n")
        for i, (doc, score) in enumerate(results):
            f.write(f"### {i+1}. {doc.metadata.get('title')} (score: {score:.4f})\n")
            f.write(f"- タグ: {doc.metadata.get('tags')}\n")
            f.write(f"\n{doc.page_content}\n\n")
        f.write("## 🧠 ChatGPTの回答\n\n")
        f.write(answer)

    # HTML保存
    html_path = base.with_suffix(".html")
    with open(html_path, "w", encoding="utf-8") as f:
        html = markdown.markdown(Path(md_path).read_text(), extensions=["extra", "tables"])
        f.write(f"<html><body>{html}</body></html>")

# CLIエントリーポイント
def main():
    parser = argparse.ArgumentParser(description="RAG CLI with JSON/Markdown knowledge and tag filtering")
    parser.add_argument("--query", type=str, required=True, help="質問内容")
    parser.add_argument("--tags", nargs="*", help="タグでフィルタする（スペース区切り）")
    args = parser.parse_args()

    print("🔍 ナレッジを読み込み中...")
    knowledge = load_knowledge("knowledge", tag_filter=args.tags)
    print(f"✅ {len(knowledge)} 件のナレッジを読み込みました")

    db = create_vectorstore(knowledge)

    print("🔍 類似ナレッジを検索中...")
    results = search_knowledge(db, args.query)

    print("\n📚 類似ナレッジ:")
    for i, (doc, score) in enumerate(results):
        print(f"--- {i+1} ---\nタイトル: {doc.metadata.get('title')}\nタグ: {doc.metadata.get('tags')}\nスコア: {score:.4f}\n内容: {doc.page_content}\n")

    print("🧠 ChatGPTによる回答:")
    answer = ask_gpt(results, args.query)
    print("\n" + answer)

    save_report_md_html(args.query, results, answer)
    print("\n📝 結果をMarkdownとHTMLで保存しました")

if __name__ == "__main__":
    main()
