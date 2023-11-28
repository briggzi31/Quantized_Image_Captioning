from datasets import Dataset


class ImageCaptioningDataset(Dataset):
    

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        # encoding = {'labels' if k == 'input_ids' else k:v.squeeze() for k, v in encoding.items()}

        new_encoding = {}
        for k, v in encoding.items():
            v = v.squeeze()
            if k == 'input_ids':
                new_encoding['labels'] = v
            else:
                new_encoding[k] = v

        return new_encoding
