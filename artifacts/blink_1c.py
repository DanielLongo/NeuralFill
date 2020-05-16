import torch
import numpy as np
from torch.utils import data
from artifact_utils import get_filenames_by_artifacts
import sys 
import h5py

sys.path.append("../")
from utils import custom_norm_batch


class Blink1c(data.Dataset):

	def __init__(self, max_num_examples, length=1000):

		self.max_num_examples = max_num_examples
		self.length = length
		self.examples = []


		self.load_examples()

	def load_examples(self):
		artifact_files = self.get_blinks()["blink"]

		for file, pos in artifact_files:

			if len(self.examples) == self.max_num_examples:
				break

			hdf = h5py.File(file)
			rec = hdf['record-0']
			signals = rec['signals']
			
			self.examples.append((signals[26:27, pos - 100: pos + (self.length - 100)]))

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, index):
		cur_tensor = torch.from_numpy(self.examples[index]).type('torch.FloatTensor')
		return self.signal_preprocess(cur_tensor)
		# cur_tensor = custom_norm_batch(cur_tensor)
		# cur_tensor = torch.nn.functional.relu(cur_tensor - .5) * 2 # remove noise and only keep blinks
		# return cur_tensor

	def signal_preprocess(self, sample):
		max_val = torch.max(sample)
		min_val = torch.min(sample)
		sample[abs(sample) < abs(min_val)/2] = 0
		sample = custom_norm_batch(sample)
		return sample

	def get_blinks(self):
		target_artifacts = {
			"blink" : self.max_num_examples,
		}
		artifacts = get_filenames_by_artifacts(target_artifacts)
		return artifacts


if __name__ == "__main__":
	dataset = Blink1c(1000)
	print("Length", len(dataset))
	print("Sample Shape", dataset[0].shape)

