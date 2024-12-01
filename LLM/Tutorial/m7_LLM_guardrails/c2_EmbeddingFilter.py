import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingFilter:
    def __init__(self, model_name='distiluse-base-multilingual-cased-v1'):
        self.model = SentenceTransformer(model_name)
        self.sensitive_embeddings = {}
    
    def add_sensitive_words(self, words):
        embeddings = self.model.encode(words)
        for word, emb in zip(words, embeddings):
            self.sensitive_embeddings[word] = emb
    
    def filter(self, text, threshold=0.7):
        text_emb = self.model.encode([text])[0]
        for word, emb in self.sensitive_embeddings.items():
            sim = self.cosine_similarity(text_emb, emb)
            if sim >= threshold:
                return True, word
        return False, None
    
    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    ef = EmbeddingFilter()
    sensitive_words = ["黄色小说", "暴力行为", "不良信息"]
    ef.add_sensitive_words(sensitive_words)
    text = "这是一篇包含不良内容的文章。"
    is_sensitive, word = ef.filter(text)
    if is_sensitive:
        print(f"检测到敏感词：{word}")
    else:
        print("未检测到敏感内容。")
