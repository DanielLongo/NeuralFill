import csv
import random
from data_utils import get_eeghdf_files

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



if __name__ == "__main__":
	# generate_csv_from_dir("./dataset_csv/sample_file.csv", "/mnt/data1/eegdbs/SEC-0.1/stanford/")
	generate_splits_csvs_from_dir("./dataset_csv/sec-.1", "/mnt/data1/eegdbs/SEC-0.1/stanford/")




