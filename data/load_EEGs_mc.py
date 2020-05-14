import torch
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils import data
from torch.nn import functional as F
from data_utils import load_eeg_file, get_recordings_from_csv

import sys
sys.path.append("../")
from utils import custom_norm_batch

class EEGDatasetMc(data.Dataset):

	def __init__(self, files_csv, select_channels=[0,1,3], max_num_examples=-1, length=1000, target_freq=200):

		# since all data is only one channel only takes the selected channel
		self.select_channels = select_channels
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
		all_signals, specs = load_eeg_file(filename)

		if (self.target_freq != -1) and (int(specs["sample_frequency"]) != self.target_freq):
			return None

		signals = []
		for channel in self.select_channels:
			signals.append(all_signals[channel])
		signals = np.stack(signals)
		# signals = np.expand_dims(signals, axis=0).T

		if signals.shape[1] < (self.length  + start_pos):
			return None


		start_pos = int(start_pos)
		signals = signals[:, start_pos:start_pos + self.length]
		return signals

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, index):
		cur_tensor = torch.from_numpy(self.examples[index]).type('torch.FloatTensor')
		return custom_norm_batch(cur_tensor)


if __name__ == "__main__":
	files_csv = "./dataset_csv/sample_file.csv"
	dataset = EEGDatasetMc(files_csv, max_num_examples=10)
	print("Length", len(dataset))
	print("Sample Shape", dataset[0].shape)
