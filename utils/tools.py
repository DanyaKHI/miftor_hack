import pandas as pd
import configs as cfg
import json
import spacy


class Heuristic_I:
    def __init__(self):
        with open(cfg.I_value_target_words, 'r') as json_file:
            self.target_words = set(json.load(json_file))
        self.nlp = spacy.load('ru_core_news_sm')

    def inference(self, texts, b_values, future_look=10):
            result = {
                'text': texts,
                'b_values': b_values,
                'i_values': []
            }
            for text, b_value_list in zip(texts, b_values):
                text = text.split()
                i_values = []
                for val_idx in b_value_list:
                    idx = val_idx
                    while idx + 1 < len(text) and idx + 1 < val_idx + future_look:
                        if self.nlp(text[idx + 1])[0].pos_ == 'NUM' or (text[idx + 1] in self.target_words):
                            break
                        idx += 1

                    while idx + 1 < len(text) and (self.nlp(text[idx + 1])[0].pos_ == 'NUM' or (text[idx + 1] in self.target_words)):
                        i_values.append(idx + 1)
                        idx += 1

                result['i_values'].append(i_values)
            return pd.DataFrame(result)
