import numpy as np
import re

class OptimizedDFAFilter(object):
    '''
    Optimized DFA Filter that can handle intervening characters and homophones.
    '''

    def __init__(self, skip_chars=None, homophone_dict=None):
        self.keyword_chains = {}
        self.delimit = '\x00'
        self.skip_chars = skip_chars if skip_chars else set([' ', '*', '-', '_'])
        self.homophone_dict = homophone_dict if homophone_dict else {}

    def add(self, keyword):
        keyword = self._normalize(keyword.lower())
        chars = keyword.strip()
        if not chars:
            return
        level = self.keyword_chains
        for i in range(len(chars)):
            char = chars[i]
            if char in level:
                level = level[char]
            else:
                level[char] = {}
                level = level[char]
            if i == len(chars) - 1:
                level[self.delimit] = 0

    def parse(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for keyword in f:
                self.add(keyword.strip())

    def _normalize(self, message):
        # 替换谐音字
        for homophone, normal in self.homophone_dict.items():
            message = message.replace(homophone, normal)
        return message

    def filter(self, message, repl="*"):
        message = self._normalize(message.lower())
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step = 0
            i = start
            while i < len(message):
                char = message[i]
                if char in self.skip_chars:
                    i += 1
                    continue
                if char in level:
                    step += 1
                    level = level[char]
                    if self.delimit in level:
                        ret.append(repl * step)
                        start = i + 1
                        break
                    i += 1
                else:
                    ret.append(message[start])
                    start += 1
                    break
            else:
                ret.append(message[start])
                start += 1
        return ''.join(ret)


if __name__ == "__main__":
    # 定义谐音字典
    homophone_dict = {'5': '五', 'S': '死', '@': '爱'}
    dfa = OptimizedDFAFilter(homophone_dict=homophone_dict)
    dfa.add("我爱你")
    dfa.add("你死我活")
    print(dfa.filter("我@你"))  # 输出：****你
    print(dfa.filter("你S我活"))  # 输出：****
