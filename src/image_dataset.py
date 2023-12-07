# from datasets import Dataset
import torch
from torch.utils.data import Dataset


class ImageCaptioningDataset(Dataset):
    

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

        # super().__init__(dataset)
    

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")

        new_encoding = {k: v.squeeze() for k, v in encoding.items()}
        new_encoding['text'] = item['text']
        return new_encoding
    

    def __str__(self):
        out = str(self.dataset) + "\n"
        out += str(self.processor)
        return out


    def collate_fn(self, batch):
        processed_batch = {}

        columns_names = batch[0].keys()
        for col in columns_names:
            if col == 'pixel_values':
                processed_batch[col] = torch.stack([x[col] for x in batch])
            elif col == 'text':
                text_inputs = self.processor.tokenizer(
                    [x["text"] for x in batch], padding=True, return_tensors="pt"
                )
                
                processed_batch['labels'] = text_inputs['input_ids']
                processed_batch['attention_mask'] = text_inputs['attention_mask']

        # bos = self.processor.tokenizer.bos_token
        # bos_list = [bos for _ in range(len(batch))]
        # # bos_list = [bos]
        # # processed_batch['bos'] = self.processor.tokenizer(bos_list, padding=True, return_tensors="pt")["input_ids"]
        # processed_batch['bos'] = torch.tensor([[2], [2]], dtype=torch.int)

        return processed_batch
