"""Interactive prediction CLI."""

from __future__ import annotations

from typing import Any

from .inference import load_model, load_texts_from_file, predict_batch, predict_text


def interactive_mode(
    model: Any,
    tfidf_vectorizer: Any,
    embedding_model: Any,
    emb_weight: float,
) -> None:
    """Run the interactive prediction loop.

    Args:
        model: Trained classifier.
        tfidf_vectorizer: TF-IDF vectorizer.
        embedding_model: Sentence embedding model.
        emb_weight: Embedding feature weight.
    """
    print("\n" + "=" * 60)
    print("交互式广告检测")
    print("=" * 60)
    print("命令:")
    print("  - 直接输入文本进行检测")
    print("  - 输入 'batch' 进入批量模式")
    print("  - 输入 'file <文件路径>' 从文件读取")
    print("  - 输入 'q' 或 'quit' 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("💬 请输入: ").strip()

            if user_input.lower() in ["q", "quit", "exit"]:
                print("\n👋 再见！")
                break

            if not user_input:
                print("⚠️  输入不能为空\n")
                continue

            if user_input.lower() == "batch":
                print("\n进入批量模式（输入空行结束）:")
                batch_texts = []
                while True:
                    line = input(f"  [{len(batch_texts) + 1}] ").strip()
                    if not line:
                        break
                    batch_texts.append(line)

                if batch_texts:
                    results = predict_batch(
                        batch_texts,
                        model,
                        tfidf_vectorizer,
                        embedding_model,
                        emb_weight,
                    )
                    print("\n" + "-" * 60)
                    print("批量预测结果:")
                    print("-" * 60)
                    for i, (text, result, conf) in enumerate(results, 1):
                        conf_str = f"(置信度: {conf:.3f})" if conf else ""
                        print(f"{i}. [{result}] {conf_str}")
                        print(f"   {text[:60]}{'...' if len(text) > 60 else ''}")
                    print("-" * 60 + "\n")
                continue

            if user_input.lower().startswith("file "):
                file_path = user_input[5:].strip()
                texts = load_texts_from_file(file_path)
                if texts:
                    results = predict_batch(
                        texts,
                        model,
                        tfidf_vectorizer,
                        embedding_model,
                        emb_weight,
                    )
                    print("\n" + "-" * 60)
                    for i, (text, result, conf) in enumerate(results, 1):
                        conf_str = f"(置信度: {conf:.3f})" if conf else ""
                        print(f"{i}. [{result}] {conf_str}")
                        print(f"   {text[:60]}{'...' if len(text) > 60 else ''}")
                    print("-" * 60)

                    ad_count = sum(1 for _, r, _ in results if r == "广告")
                    print(f"\n统计: 广告 {ad_count} 条 | 正常 {len(results) - ad_count} 条\n")
                continue

            result, confidence = predict_text(
                user_input,
                model,
                tfidf_vectorizer,
                embedding_model,
                emb_weight=emb_weight,
            )

            print(f"预测结果: {result}")
            if confidence is not None:
                print(f"置信度: {confidence:.3f}")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n程序已中断，再见！")
            break
        except Exception as exc:
            print(f"发生错误: {exc}\n")


def run_interactive() -> None:
    """Entry point for interactive prediction."""
    print("广告检测系统")
    print("=" * 60)

    model, tfidf_vectorizer, embedding_model, emb_weight = load_model()

    if not all([model, tfidf_vectorizer, embedding_model, emb_weight]):
        return

    interactive_mode(model, tfidf_vectorizer, embedding_model, emb_weight)


if __name__ == "__main__":
    run_interactive()
