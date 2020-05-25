import csv
import random
from data_utils import get_eeghdf_files
import sys
sys.path.append("../")
from constants import *
import pandas as pd
from collections import Counter

def generate_csv_from_dir(save_filename, dir_path, start_pos=1e3*20):
	files = get_eeghdf_files(dir_path)
	headers = [["filename", "start_pos"]]
	recordings = [[file, start_pos] for file in files]
	print("len recordings", len(recordings))

	with open(save_filename, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(headers + recordings)

	return recordings


def generate_splits_csvs_from_dir(save_filename, dir_path, split=[.8, .1, .1], start_pos=1e3*20):
	assert(sum(split) == 1), "split sum should be 1 so that all examples are included"
	recordings = generate_csv_from_dir(save_filename, dir_path, start_pos=start_pos)
	random.shuffle(recordings)
	
	num_train = int(len(recordings) * split[0])
	num_dev = int(len(recordings) * split[1])
	num_test = int(len(recordings) * split[2])

	examples_train = recordings[:num_train]
	examples_dev = recordings[num_train: num_train + num_dev]
	examples_test = recordings[num_train + num_dev: num_train + num_dev + num_test]

	headers = [["filename", "start_pos"]]

	with open(save_filename + "_train.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(headers + examples_train)

	with open(save_filename + "_dev.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(headers + examples_dev)

	with open(save_filename + "_test.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(headers + examples_test)

def generate_norm_csv(save_filename, filenames, recordings_df):
	recordings_df = recordings_df.loc[recordings_df['filename'].isin(filenames)]

	header = [["filename", "start_pos"]]
	out = []
	
	for _, row in recordings_df.iterrows():
		filename = row['filename']
		start_pos = int(row['pos'])

		# each recording is good for 10s so makes 3 x 3ish sec long recordings
		out.append([filename, start_pos])
		out.append([filename, start_pos + 784])
		out.append([filename, start_pos + 784 * 2])

	with open(save_filename, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(header + out)

def generate_norm_csvs(split=[.8, .1, .1]):

	assert(sum(split) == 1), "split sum should be 1 so that all examples are included"
	recordings = pd.read_csv(ALL_NORMAL_FILES_CSV)
	source_files = list(recordings.filename)
	m = len(source_files)
	m_train = int(split[0] * m)
	m_dev = int(split[1] * m)
	m_test = int(split[2] * m)

	print("Desired Split", "Train:", m_train, "Dev:", m_dev, "Test:", m_test)

	source_files_count_dict = Counter(source_files)

	# shuffle the dict
	l = list(source_files_count_dict.items())
	random.shuffle(l)
	source_files_count_dict = dict(l)

	train_filenames, dev_filenames, test_filenames = [], [], []
	train_count, dev_count, test_count = 0, 0, 0

	for filename, count in source_files_count_dict.items():

		# first part ensures that adding the examples will not exceed the desired m for test
		# second part ensures that no one file comprises more than 1/3 of the examples for test
		if test_count + count <= m_test and (count * 3) < m_test:
			test_filenames.append(filename)
			test_count += count

		elif dev_count + count <= m_dev and (count * 3) < m_dev:
			dev_filenames.append(filename)
			dev_count += count

		else:
			train_filenames.append(filename)
			train_count += count

	print("Actual Split",  "Train:", train_count, "Dev:", dev_count, "Test:", test_count)

	generate_norm_csv("dataset_csv/normal_test.csv", test_filenames, recordings)
	generate_norm_csv("dataset_csv/normal_dev.csv", dev_filenames, recordings)
	generate_norm_csv("dataset_csv/normal_train.csv", train_filenames, recordings)




if __name__ == "__main__":
	# generate_csv_from_dir("./dataset_csv/sample_file.csv", "/mnt/data1/eegdbs/SEC-0.1/stanford/")
	# generate_splits_csvs_from_dir("./dataset_csv/sec-.1", "/mnt/data1/eegdbs/SEC-0.1/stanford/")
	generate_norm_csvs()




