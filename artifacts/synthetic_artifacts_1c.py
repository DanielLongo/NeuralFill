import torch
import numpy as np
from torch.utils import data

class SyntheticArtifiacts1c(data.Dataset):

	def __init__(self, num_examples, length=1000):

		self.num_examples = num_examples
		self.length = length
		self.examples = []

		self.load_examples()

	def load_examples(self):

		while len(self.examples) != self.num_examples:
			
			# 60 hz noise
			self.examples.append(gen_sin(60, self.length))

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, index):
		cur_tensor = torch.from_numpy(self.examples[index])
		return cur_tensor


def gen_sin(fs, length):
	f0 = 1
	t = np.arange(length)
	sinusoid = np.sin(2*np.pi*t*(f0/fs))
	sinusoid = np.expand_dims(sinusoid, axis=0).T
	return sinusoid

if __name__ == "__main__":
	dataset = SyntheticArtifiacts1c(10)
	print("Length", len(dataset))
	print("Sample Shape", dataset[0].shape)

