# RAG Test Kit

探索的テストの観点生成を支援する Retrieval-Augmented Generation (RAG) CLI ツール。

## 機能

- JSON / Markdown ナレッジの読み込み
- ChatGPT（gpt-3.5-turbo）による観点生成
- タグフィルタ、検索スコア表示
- Markdown / HTML 出力


## 使い方

```
python main.py --query "ログインフォームの異常系テスト" --tags バリデーション 入力系
```

## 出力例
output/ログインフォームの異常系テスト.md
output/ログインフォームの異常系テスト.html
