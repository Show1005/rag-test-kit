# RAG Test Kit

探索的テストの観点生成を支援する Retrieval-Augmented Generation (RAG) CLI ツール。

## 機能

- JSON / Markdown ナレッジの読み込み
- ChatGPT（gpt-3.5-turbo）による観点生成
- タグフィルタ、検索スコア表示
- Markdown / HTML 出力


## 使い方
1. `.env.example` を `.env` にコピーし、OpenAI API キーを設定します。
```
cp .env.example .env
# .env を開いて OPENAI_API_KEY に自身のキーを設定
```

```
streamlit run app.py
```

CLI での利用例:
```
python main.py --query "ログインフォームの異常系テスト"
```

## 出力例
output/ログインフォームの異常系テスト.md
output/ログインフォームの異常系テスト.html

![image](https://github.com/user-attachments/assets/36321869-b118-4af3-9a6c-ec9e75108144)
