import gzip
import shutil

def decompress_gz_file(input_file_path, output_file_path):
    with gzip.open(input_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":
    # Example usage
    input_file_path = 'datasets/raw/sepsis/sepsis.xes.gz'
    output_file_path = 'datasets/raw/sepsis/sepsis.xes'
    decompress_gz_file(input_file_path, output_file_path)

