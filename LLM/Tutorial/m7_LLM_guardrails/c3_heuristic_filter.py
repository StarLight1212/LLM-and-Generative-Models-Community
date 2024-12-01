import jieba
from collections import Counter

class HeuristicFilter:
    def __init__(self):
        self.sensitive_words = set()
    
    def load_sensitive_words(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for word in f:
                self.sensitive_words.add(word.strip())
    
    def filter(self, text, frequency_threshold=2):
        words = jieba.lcut(text)
        word_counts = Counter(words)
        sensitive_count = sum([word_counts[word] for word in self.sensitive_words if word in word_counts])
        if sensitive_count >= frequency_threshold:
            return True
        else:
            return False


if __name__ == "__main__":
    hf = HeuristicFilter()
    hf.load_sensitive_words("sensitive_words.txt")
    text = "这个电影包含大量暴力和不良内容，不适合儿童观看。"
    if hf.filter(text):
        print("检测到不良内容。")
    else:
        print("内容正常。")
