# CUDA is required to run all these files!

## python: 3.9.16, libraries are listed in files.

## Models we are using for a final version:

SimpleViT_GPU57 -> Non pre-trained ViT model with Data Augmentation

Pre-trained ViT -> Pre-trained ViT model with Data Augmentation

## Other files are also included for reference purpose.

### All documents are equipped with W&B tracking. If you do not want to use W&B, you can comment out the 'wandb' related code.

### If you want to run and test the code, you need to change the path of the file corresponding to the beginning of the code, i.e. DATA_FOLDER_PATH = '[YOUR_REPOSITORIE]/Animals_with_Attributes2/', JPEGIMAGES_FOLDER_PATH = '[YOUR_REPOSITORIE]/Animals_with_Attributes2/JPEGImages/'