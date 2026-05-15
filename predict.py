import joblib
import os
import time
from scipy.sparse import hstack, csr_matrix


def load_model():
    """加载训练好的模型和向量化器 + embedding 模型"""
    try:
        print("正在加载模型...")
        start_time = time.time()

        model = joblib.load("ad_model.pkl")
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        embedding_model = joblib.load("embedding_model.pkl")
        emb_weight = joblib.load("emb_weight.pkl")

        load_time = time.time() - start_time
        print(f"✓ 模型加载成功！（耗时 {load_time:.2f}秒）")
        return model, tfidf_vectorizer, embedding_model, emb_weight

    except FileNotFoundError as e:
        print("❌ 错误：找不到模型文件！")
        print("请先运行 main.py 训练并保存模型。")
        return None, None, None, None
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        return None, None, None, None


def predict_text(
    text, model, tfidf_vectorizer, embedding_model, show_confidence=True, emb_weight=1.0
):
    """
    对单条文本进行预测（融合特征）

    参数:
        text: 待预测文本
        model: 训练好的分类模型
        tfidf_vectorizer: TF-IDF向量化器
        embedding_model: 句子嵌入模型
        show_confidence: 是否显示置信度
        emb_weight: embedding 特征权重（与训练时一致）

    返回:
        result: 预测结果字符串
        confidence: 置信度分数（如果可用）
    """
    # TF-IDF特征提取
    tfidf_vec = tfidf_vectorizer.transform([text])

    # 句子嵌入特征提取
    emb_vec = embedding_model.encode([text], normalize_embeddings=True)
    emb_vec_sparse = csr_matrix(emb_vec)  # 转换为稀疏矩阵

    # 特征融合（乘以 emb_weight 与训练时保持一致）
    final_vec = hstack([tfidf_vec, emb_vec_sparse * emb_weight])

    # 预测
    prediction = model.predict(final_vec)[0]
    result = "广告" if prediction == 1 else "正常"

    # 获取置信度
    confidence = None
    if show_confidence and hasattr(model, "decision_function"):
        score = model.decision_function(final_vec)[0]
        confidence = abs(score)

    return result, confidence


def predict_batch(text_list, model, tfidf_vectorizer, embedding_model, emb_weight=1.0):
    """
    批量预测多条文本

    参数:
        text_list: 文本列表
        model: 训练好的分类模型
        tfidf_vectorizer: TF-IDF向量化器
        embedding_model: 句子嵌入模型
        emb_weight: embedding 特征权重（与训练时一致）

    返回:
        results: 预测结果列表
    """
    if not text_list:
        return []

    print(f"\n正在批量预测 {len(text_list)} 条文本...")
    start_time = time.time()

    # TF-IDF特征提取
    tfidf_vec = tfidf_vectorizer.transform(text_list)

    # 句子嵌入特征提取
    emb_vec = embedding_model.encode(
        text_list, normalize_embeddings=True, show_progress_bar=True
    )
    emb_vec_sparse = csr_matrix(emb_vec)

    # 特征融合（乘以 emb_weight 与训练时保持一致）
    final_vec = hstack([tfidf_vec, emb_vec_sparse * emb_weight])

    # 批量预测
    predictions = model.predict(final_vec)

    # 获取置信度
    confidences = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(final_vec)
        confidences = [abs(s) for s in scores]

    # 组装结果
    results = []
    for i, (text, pred) in enumerate(zip(text_list, predictions)):
        result = "广告" if pred == 1 else "正常"
        conf = confidences[i] if confidences else None
        results.append((text, result, conf))

    elapsed = time.time() - start_time
    print(
        f"✓ 批量预测完成！（耗时 {elapsed:.2f}秒，平均 {elapsed/len(text_list):.3f}秒/条）"
    )

    return results


def load_texts_from_file(file_path):
    """从文件加载文本，每行一条"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"✓ 从 {file_path} 读取了 {len(texts)} 条文本")
        return texts
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
        return []
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return []


def interactive_mode(model, tfidf_vectorizer, embedding_model, emb_weight):
    """交互式预测模式"""
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

            # 批量模式
            if user_input.lower() == "batch":
                print("\n进入批量模式（输入空行结束）:")
                batch_texts = []
                while True:
                    line = input(f"  [{len(batch_texts)+1}] ").strip()
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

            # 从文件读取
            if user_input.lower().startswith("file "):
                file_path = user_input[5:].strip()
                texts = load_texts_from_file(file_path)
                if texts:
                    results = predict_batch(
                        texts, model, tfidf_vectorizer, embedding_model, emb_weight
                    )
                    print("\n" + "-" * 60)
                    for i, (text, result, conf) in enumerate(results, 1):
                        conf_str = f"(置信度: {conf:.3f})" if conf else ""
                        print(f"{i}. [{result}] {conf_str}")
                        print(f"   {text[:60]}{'...' if len(text) > 60 else ''}")
                    print("-" * 60)

                    # 统计
                    ad_count = sum(1 for _, r, _ in results if r == "广告")
                    print(
                        f"\n统计: 广告 {ad_count} 条 | 正常 {len(results)-ad_count} 条\n"
                    )
                continue

            # 单条预测
            result, confidence = predict_text(
                user_input,
                model,
                tfidf_vectorizer,
                embedding_model,
                emb_weight=emb_weight,
            )

            # 显示结果
            print(f"预测结果: {result}")
            if confidence is not None:
                print(f"置信度: {confidence:.3f}")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n程序已中断，再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}\n")


def main():
    print("广告检测系统")
    print("=" * 60)

    # 加载模型
    model, tfidf_vectorizer, embedding_model, emb_weight = load_model()

    if not all([model, tfidf_vectorizer, embedding_model, emb_weight]):
        return

    # 启动交互模式
    interactive_mode(model, tfidf_vectorizer, embedding_model, emb_weight)


if __name__ == "__main__":
    main()
