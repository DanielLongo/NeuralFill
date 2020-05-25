import torch
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils import data
from torch.nn import functional as F
from data_utils import load_eeg_file, get_recordings_from_csv

class EEGDataset1c(data.Dataset):

	def __init__(self, files_csv, max_num_examples=-1, length=1000, target_freq=200):

		# since all data is only one channel only takes the selected channel
		self.select_channel = 10 #0
		self.target_freq = target_freq # -1 for everything
		self.length = length
		self.max_num_examples = max_num_examples
		self.examples = []

		self.recordings = get_recordings_from_csv(files_csv)

		self.load_examples()

	def load_examples(self):
		for recording in self.recordings:

			cur_filename = recording["filename"]
			cur_start_pos = float(recording["start_pos"])

			if len(self.examples) == self.max_num_examples:
				break
			
			cur_signals = self.load_signals(cur_filename, cur_start_pos)
			if cur_signals is None:				
				continue
			self.examples.append(cur_signals)


	def load_signals(self, filename, start_pos):
		signals, specs = load_eeg_file(filename)

		if (self.target_freq != -1) and (int(specs["sample_frequency"]) != self.target_freq):
			return None

		signals = signals[self.select_channel]
		signals = np.expand_dims(signals, axis=0)

		if signals.shape[1] < (self.length  + start_pos):
			return None


		start_pos = int(start_pos)
		signals = signals[:, start_pos:start_pos + self.length]
		return signals

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, index):
		cur_tensor = torch.from_numpy(self.examples[index]).type('torch.FloatTensor')

		# tensor values must be between 0 and 1
		# 1e3
		# print('(torch.abs(cur_tensor).mean())', (torch.abs(cur_tensor).mean()))
		abs_mean = (torch.abs(cur_tensor).mean())
		if abs_mean == 0:
			return cur_tensor
		cur_tensor = cur_tensor/(abs_mean)
		cur_tensor = (torch.tanh(cur_tensor) + 1)/2

		# cur_tensor = (F.tanh(cur_tensor/5e2) + 1)/2
		return cur_tensor

if __name__ == "__main__":
	files_csv = "./dataset_csv/sample_file.csv"
	dataset = EEGDataset1c(files_csv, max_num_examples=100*10*2)
	print("Length", len(dataset))
	print("Sample Shape", dataset[0].shape)
