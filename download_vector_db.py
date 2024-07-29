import os
import gdown

# URL of the Google Drive folder
folder_url = 'https://drive.google.com/drive/folders/1RhG1w_VVY938ogHRhZ7K5Tapnvs5b-PW?usp=share_link'

# Use gdown to download the folder
output = './downloaded_folder'
if not os.path.exists(output):
    os.makedirs(output)
gdown.download_folder(folder_url, output=output, quiet=False, use_cookies=False)
