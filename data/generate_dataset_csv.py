import csv
from data_utils import get_eeghdf_files

def generate_csv_from_dir(save_filename, dir_path, start_pos=1e3*20):
	files = get_eeghdf_files(dir_path)
	headers = [["filename", "start_pos"]]
	recordings = [[file, start_pos] for file in files]

	with open(save_filename, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(headers + recordings)


if __name__ == "__main__":
	generate_csv_from_dir("./dataset_csv/sample_file.csv", "/mnt/data1/eegdbs/SEC-0.1/stanford/")




