import zipfile
import os

def unzip_all_in_directory(source_dir, target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate through all files in the source directory
    for item in os.listdir(source_dir):
        if item.endswith('.zip'):
            zip_path = os.path.join(source_dir, item)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f'Unzipped {zip_path} to {target_dir}')

# Example usage:
source_directory = '/share/edc/home/antonis/moatless-tools/_20240522-voyage-code-2'
target_directory = '/share/edc/home/antonis/moatless-tools/20240522-voyage-code-2'

unzip_all_in_directory(source_directory, target_directory)