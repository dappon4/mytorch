from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import transformers
import os

tokenizer_jp = transformers.AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

#print(tokenizer_jp.encode("トンネルを抜けるとそこはそこはパラダイスだった。空には虹がかかり、その先にはシュガーランドが広がっていた。"))

with open("../../dataset/JPEN_corpus/en_ja/en-ja.bicleaner05.txt", "r") as f:
    line = f.readline()