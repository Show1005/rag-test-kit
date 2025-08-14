# RAG Test Kit

探索的テストの観点生成を支援する Retrieval-Augmented Generation (RAG) CLI ツール。

## 機能

- JSON / Markdown ナレッジの読み込み
- ChatGPT（gpt-5-nano）による観点生成
- タグフィルタ、検索スコア表示
- Markdown / HTML 出力
- サイドバーで検索履歴を表示
- 類似ナレッジは折りたたみカードで表示し、UIを改善


## 使い方
.envに以下設定
```
OPENAI_API_KEY=xxxxxxxxx(openAiのapiキーを取得して設定してください(課金しないと使えません))
```

```
streamlit run app.py
```

## 出力例
output/ログインフォームの異常系テスト.md
output/ログインフォームの異常系テスト.html

![image](https://github.com/user-attachments/assets/36321869-b118-4af3-9a6c-ec9e75108144)
