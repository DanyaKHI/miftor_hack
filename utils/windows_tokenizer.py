import numba


class WindowsTokenizer:
    token_dict: dict
    target_words: list

    def __init__(self, tokens: dict, target_words: set):
        self.token_dict = numba.typed.Dict.empty(
            key_type=numba.types.unicode_type,
            value_type=numba.types.float64
        )

        for word, token in tokens.items():
            self.token_dict[word] = token

        self.target_words = numba.typed.Dict.empty(
            key_type=numba.types.unicode_type,
            value_type=numba.types.boolean
        )
        for tgt_word in target_words:
            self.target_words[tgt_word] = True

    @staticmethod
    # @njit
    def __process_sentence(arr_sent: list, targets: list, tokens: numba.typed.Dict,
                           target_words: list, radius: int, indexing: int, is_fit: bool) -> tuple:

        padding = ['PAD'] * radius
        arr_sent = padding + arr_sent + padding

        arr_samples = []
        arr_targets = []
        arr_indicies = numba.typed.List()

        for i in range(2 * radius + 1, len(arr_sent) + 1):
            if arr_sent[i - radius - 1] not in target_words:
                continue

            c_samp = arr_sent[i - 2 * radius - 1: i]
            t_samp = [tokens.get(word, 1) for word in c_samp]

            arr_samples.append(t_samp)
            arr_indicies.append((indexing, i - 2 * radius - 1))

            if is_fit:
                if (i - 2 * radius - 1) in targets:
                    arr_targets.append(1)
                else:
                    arr_targets.append(0)

        return arr_samples, arr_targets, arr_indicies

    def __processing(self, X: list, y: list, radius: int, is_fit: bool):

        arr_samples = []
        arr_targets = []
        arr_indicies = []

        for i in range(len(X)):
            sample = X[i]
            tgt = y[i] if is_fit else []

            # print(self.token_dict)
            # print(self.target_words)
            c_samp, c_tgt, c_ind = self.__process_sentence(
                arr_sent=sample.split(),
                targets=tgt,
                tokens=self.token_dict,
                target_words=self.target_words,
                radius=radius,
                indexing=i,
                is_fit=is_fit)

            arr_samples.extend(c_samp)
            arr_targets.extend(c_tgt)
            arr_indicies.extend(c_ind)

        return arr_samples, arr_targets, arr_indicies

    def fit_transform(self, X, y, radius=5):
        return self.__processing(X=list(X),
                                 y=list(y),
                                 radius=radius,
                                 is_fit=True)

    def transform(self, X, radius=5):
        res = self.__processing(X=list(X),
                                y=None,
                                radius=radius,
                                is_fit=False)
        return res[0], res[2]
