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
        # encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")

        # encoding = {'labels' if k == 'input_ids' else k:v.squeeze() for k, v in encoding.items()}

        print("encoding", encoding)

        # new_encoding = {}
        # for k, v in encoding.items():
        #     v = v.squeeze()
        #     if k == 'input_ids':
        #         new_encoding['labels'] = v
        #     else:
        #         new_encoding[k] = v
        new_encoding = {k: v.squeeze() for k, v in encoding.items()}
        new_encoding['text'] = item['text']

        print("\n get_item")
        print("new_encoding", new_encoding)
        return new_encoding
    

    def __str__(self):
        out = str(self.dataset) + "\n"
        out += str(self.processor)
        return out

    
    @staticmethod
    def collate_fn(batch):
        print("\nINSIDE COLLATE_FN")
        print("batch", batch)
        processed_batch = {}

        columns_names = batch[0].keys()
        for col in columns_names:
            if col == 'pixel_values':
                processed_batch[col] = torch.stack([x[col] for x in batch])
            elif col == 'text':
                text_inputs = processor.tokenizer(
                    [x["text"] for x in batch], padding=True, return_tensors="pt"
                )
                processed_batch['labels'] = text_inputs['token_ids']
                processed_batch['attention_mask'] = text_inputs['attention_mask']

            # processed_batch[col] = torch.stack([x[col] for x in batch])

        return processed_batch
