from ruBert import ruBERT
from utils.linear_models import LinearModel
import configs as cfg
from utils.lemmatizer import lemmatize
from datetime import datetime
from utils.tools import Heuristic_I


class PIPELINE:

    def __init__(self):
        self.rubert = ruBERT()
        self.b_discount = LinearModel(path_model=cfg.B_discount_model,
                                      path_tokens=cfg.B_discount_tokens,
                                      path_target_words=cfg.B_discount_target_words,
                                      radius=cfg.B_discount_RAD
                                      )

        self.b_value = LinearModel(path_model=cfg.B_value_model,
                                   path_tokens=cfg.B_value_tokens,
                                   path_target_words=cfg.B_value_target_words,
                                   radius=cfg.B_value_RAD
                                   )
        self.i_value = Heuristic_I()

    @staticmethod
    def __bdisc_samples(dfg, is_lemmatize=False):
        y_pred_is = dfg['pred']
        index_is_B = []

        for i, b in enumerate(y_pred_is):
            if b:
                index_is_B.append(i)

        X_val_np = dfg['text'].to_numpy()[index_is_B]
        if is_lemmatize:
            return X_val_np, index_is_B
        else:
            return lemmatize(X_val_np), index_is_B

    @staticmethod
    def __convert_ans(len_np, len_txt, y_pred, indicies, is_B):
        y_pred_finish_zip = [[] for i in range(len_np)]

        for i, j in zip(y_pred, indicies):
            if i == 1:
                y_pred_finish_zip[j[0]].append(j[1])

        y_pred_finish = [[] for i in range(len_txt)]
        for i, j in zip(y_pred_finish_zip, is_B):
            y_pred_finish[j] = i

        return y_pred_finish

    def __filter_b_value(self, pred_b_val, count_tokens=6):
        y_pred_finishval2 = []
        for i in pred_b_val:
            cur_arr = []
            if len(i):
                cur_arr.append(i[0])
            for j in range(1, len(i)):
                if (i[j] - i[j - 1]) >= count_tokens:
                    cur_arr.append(i[j])
            y_pred_finishval2.append(cur_arr)
        return y_pred_finishval2

    def __to_not_stat_tgt(self, length, x):
        arr = ['O' for i in range(length)]
        for i, j in x.items():
            for k in j:
                arr[k] = i
        return arr

    def __tnst(self, X, y):
        arr_tgt = []

        for i, j in zip(X, y):
            arr_tgt.append(self.__to_not_stat_tgt(len(i.split()), j))
        return arr_tgt

    def inference(self, text, is_lemmatize=False):
        print(f'ruBert start: {datetime.now()}')
        df_preds = self.rubert.inference(texts_for_inference=text)
        print(f'ruBert finish: {datetime.now()}')

        X_val_np, index_is_B = self.__bdisc_samples(df_preds, is_lemmatize=is_lemmatize)
        print(f'ruBert squezze + lemmatize complete: {datetime.now()}')

        y_pred_b, b_indicies = self.b_discount.predict(X_val_np)
        y_pred_finish = self.__convert_ans(len(X_val_np), len(df_preds), y_pred_b, b_indicies, index_is_B)
        print(f'B-discount complete: {datetime.now()}')

        y_pred_b_val, b_indicies = self.b_value.predict(X_val_np)
        y_pred_finish_val = self.__convert_ans(len(X_val_np), len(df_preds), y_pred_b_val, b_indicies, index_is_B)
        y_pred_finish_val = self.__filter_b_value(y_pred_finish_val)
        print(f'B-value complete: {datetime.now()}')

        y_pred_i_value = self.i_value.inference(text, y_pred_finish_val, future_look=1)['i_values'].values
        print(f'I-value complete: {datetime.now()}')

        y_pred_finish_STR = [{} for i in range(len(text))]

        for ind, i in enumerate(y_pred_finish):
            if len(i):
                y_pred_finish_STR[ind]['B-discount'] = i

        for ind, i in enumerate(y_pred_finish_val):
            if len(i):
                y_pred_finish_STR[ind]['B-value'] = i

        for ind, i in enumerate(y_pred_i_value):
            if len(i):
                y_pred_finish_STR[ind]['I-value'] = i

        y_pred_tnst = self.__tnst(text, y_pred_finish_STR)

        return y_pred_tnst





