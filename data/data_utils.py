import csv
import os 
import h5py
import torch

def load_eeg_file(filename):
	hdf = h5py.File(filename, "r")
	atributes = hdf["patient"].attrs
	rec = hdf["record-0"]
	# print("KJHFKJVHSKDLJHF", list(rec['signal_labels'])[:4])
	signals = rec["signals"]
	specs = {
		"sample_frequency": rec.attrs["sample_frequency"],
		"number_channels": rec.attrs["number_channels"]
	}
	return signals, specs

def get_eeghdf_files(data_dir):
	all_files = os.listdir(data_dir)
	eeghdf_files = []
	for file in all_files:
		if file.split(".")[-1] == "eeghdf":
			eeghdf_files.append(data_dir + file)
	return eeghdf_files

def get_recordings_from_csv(csv_filename):
	#  csv file is  [[filename, start], ....]
	# returns [{filename: , start: }...]

	with open(csv_filename, newline='') as f:
		reader = csv.reader(f)
		data = list(reader)
	
	data.pop(0) # first entry is headers

	out = []
	for entry in data:
		out.append({"filename": entry[0], "start_pos": entry[1]})
	return out

def normalize(x):
	# return torch.sigmoid((x + 2000)/4000)
	return (x-x.min())/(x.max()-x.min())

def scale(x, d=2000):
	x /= d
	x = .5 + ((torch.tanh(x))/2)
	return x

def standardize(x):
#	 mean = torch.mean(x.view(-1))
	mean = -6.066188 # overall mean
#	 std = torch.std(x.view(-1))
	std = 660 # overall std
	return (x-mean)/(std)