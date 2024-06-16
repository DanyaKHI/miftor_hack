import joblib
import json
from utils.windows_tokenizer import WindowsTokenizer


class LinearModel:
    def __init__(self, path_model, path_tokens, path_target_words, radius):
        self.model = joblib.load(path_model)
        self.radius = radius
        with open(path_tokens, 'r') as json_file:
            universal_dict = json.load(json_file)
        with open(path_target_words, 'r') as json_file:
            target_words = json.load(json_file)

        self.w_tokenizer = WindowsTokenizer(tokens=universal_dict, target_words=set(target_words))

    def predict(self, X_val_np):
        X_val_np_un, train_npun_ind = self.w_tokenizer.transform(X_val_np, radius=self.radius)
        result = self.model.predict(X_val_np_un)
        return result, train_npun_ind

