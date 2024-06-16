import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np
from configs import ruBert_BATCH_SIZE, ruBert_MAX_LEN, ruBert_path, ruBert_thershold
import pandas as pd


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }


class ruBERT:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = BertTokenizer.from_pretrained(ruBert_path)
        self.model = BertForSequenceClassification.from_pretrained(ruBert_path)

        self.model = self.model.to(self.device)

    @staticmethod
    def __create_data_loader(texts, tokenizer, max_len, batch_size):
        ds = InferenceDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_len=max_len
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4
        )

    def __get_predictions(self, data_loader):
        self.model.eval()
        texts = []
        predictions = []

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(outputs.logits, dim=1)
                predictions.extend(probs[:, 1].cpu().numpy())
                texts.extend(d["text"])

        return texts, predictions

    def inference(self, texts_for_inference: np.ndarray) -> pd.DataFrame:
        inference_data_loader = self.__create_data_loader(texts_for_inference, self.tokenizer, ruBert_MAX_LEN, ruBert_BATCH_SIZE)
        texts, predictions = self.__get_predictions(inference_data_loader)

        df_predictions = pd.DataFrame({
            'text': texts,
            'predicted_prob': predictions
        })

        df_predictions['pred'] = df_predictions['predicted_prob'].apply(lambda x: 1 if x > ruBert_thershold else 0)
        return df_predictions


