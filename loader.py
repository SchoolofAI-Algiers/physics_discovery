import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomTimeSeriesDataset(Dataset):
    def __init__(self, csv_file_path, transform=None):
        self.data_frame = pd.read_csv(csv_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        time_series = self.data_frame.iloc[idx, 1:50].values.astype('float32')
        fr_feature = self.data_frame.iloc[idx, -4].astype('float32')
        st_feature = self.data_frame.iloc[idx, -3].astype('float32')
        question = self.data_frame.iloc[idx, -2].astype('float32')
        answer = self.data_frame.iloc[idx, -1].astype('float32')

        sample = {
            'time_series': time_series,
            'fr': fr_feature,
            'st': st_feature,
            'question': question,
            'answer': answer
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def build_dataloader(batch_size = 32):
    # DataLoader example
    csv_file_path = 'data.csv'
    custom_dataset = CustomTimeSeriesDataset(csv_file_path)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    return data_loader

if __name__ == "__main__":
    # Assuming you have 'data.csv' in the current directory
    csv_file_path = 'data.csv'
    custom_dataset = CustomTimeSeriesDataset(csv_file_path)

    # Accessing a single sample from the dataset
    sample = custom_dataset[0]
    time_series_data = sample['time_series']
    fr_feature = sample['fr']
    st_feature = sample['st']
    question = sample['question']
    answer = sample['answer']

    print("question:", question)
    print("answer:", answer)
