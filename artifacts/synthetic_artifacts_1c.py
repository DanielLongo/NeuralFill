import torch
import numpy as np
from torch.utils import data
from mne_synthetic_artifacts import AddEOG
import random
from contextlib import contextmanager
import sys, os
import random

# https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# class SyntheticArtifiacts1c(data.Dataset):

# 	def __init__(self, num_examples, length=1000):

# 		self.num_examples = num_examples
# 		self.length = length
# 		self.examples = []

# 		self.load_examples()

# 	def load_examples(self):

# 		while len(self.examples) != self.num_examples:
			
# 			# 60 hz noise
# 			self.examples.append(gen_sin(60, self.length))

# 	def __len__(self):
# 		return len(self.examples)

# 	def __getitem__(self, index):
# 		cur_tensor = torch.from_numpy(self.examples[index]).type('torch.FloatTensor')
# 		return cur_tensor

class SyntheticArtifiactsLabeled1c(data.Dataset):

	def __init__(self, num_examples, length=1000, sample_fs=250, label=True, normalize=False, target_artifacts=None):

		self.num_examples = num_examples
		self.length = length
		self.examples = []
		self.labels = []
		self.blink_noisifier = AddEOG(4)
		self.sample_fs = sample_fs
		self.label = label
		self.normalize = normalize
		if target_artifacts is None:

			self.target_artifacts = {
			"no signal": True,
			"60hz noise": True,
			"blink": True,
			"60hz and blink": False,
		}

		else:
			self.target_artifacts = target_artifacts

		self.load_examples()


	def load_examples(self):

		zero_label, one_label, two_label, three_label = torch.zeros((10)), torch.zeros((10)), torch.zeros((10)), torch.zeros((10)) 
		zero_label[0], one_label[1], two_label[2], three_label[3] = 1, 1, 1, 1

		while len(self.examples) < self.num_examples:
			
			# No Singal
			if self.target_artifacts["no signal"]:
				self.examples.append(np.zeros((1, self.length)))
				self.labels.append(zero_label)

			# 60 hz noise
			if self.target_artifacts["60hz noise"]:
				self.examples.append(gen_sin(250, self.length).T)
				self.labels.append(one_label)

			# Eye Blink
			if self.target_artifacts["blink"]:
				self.examples.append(gen_eog(self.length, self.blink_noisifier, self.sample_fs).reshape(1, self.length))
				self.labels.append(two_label)

			if self.target_artifacts["60hz and blink"]:
				noise = gen_sin(250, self.length).T
				blink = gen_eog(self.length, self.blink_noisifier, self.sample_fs).reshape(1, self.length)
				b = random.randint(80, 120)
				combined = ((noise/b) + blink*b) / 2
				self.examples.append(combined)
				self.labels.append(three_label)


	def __len__(self):
		return len(self.examples)

	def __getitem__(self, index):
		cur_tensor = torch.from_numpy(self.examples[index]).type('torch.FloatTensor')

		if self.normalize != False:
			cur_tensor = self.normalize(cur_tensor)

		cur_tensor = cur_tensor#.numpy()
		if self.label:
			cur_label = self.labels[index]
			cur_label = cur_label#.numpy()
			return cur_tensor, cur_label
		return cur_tensor



def gen_sin(fs, length):
	# fs is sample rate
	f0 = 60
	t = np.arange(length)
	sinusoid = np.sin(2*np.pi*t*(f0/fs))
	sinusoid = np.expand_dims(sinusoid, axis=0).T
	return sinusoid

def gen_eog(length, blink_noisifier, sample_fs):
	with suppress_stdout():
		# x = np.random.randn(4, length)
		x = np.zeros((4, length))
		x = blink_noisifier.add_eog(x)
	signal = x[random.randint(0,3)]
	if np.sum(abs(signal)) == 0:
		# some signals don't have blinks
		return gen_eog(length, blink_noisifier, sample_fs)
	return signal



if __name__ == "__main__":
	dataset = SyntheticArtifiactsLabeled1c(10)
	print("Length", len(dataset))
	print("Sample Shape", dataset[0][0].shape)
	print("Sample Label", dataset[0][1])

