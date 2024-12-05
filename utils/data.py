import json
import pandas as pd

POS_TAG_COLS = ["sent_id", "word_id", "word", "lemma", "ud_tag", "penn_tag"]

def load_task():
    with open("task/task.json") as f:
        return json.load(f)

def load_pos_tags():
    with open("task/pos_tags.txt") as f:
        data = f.read().strip()
    
    d = []
    for sent_id, sent in enumerate(data.split("\n\n"), start=1):
        for word_id, tagged_word in enumerate(sent.split("\t"), start=1):
            d.append(
                {c: v for c, v in
                      zip(POS_TAG_COLS, [sent_id, word_id] + tagged_word.strip().split("\\"))
                      }
                )
    return pd.DataFrame(d)
    
if __name__ == '__main__':
    print(load_pos_tags())