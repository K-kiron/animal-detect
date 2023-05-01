# CUDA is required to run all these files!

## Environment: 
### Google colab Notebook ,python 3.10.11, libraries are listed in files.

# important packages
## Pytorch
```
pip3 install torch torchvision torchaudio
```
## EfficientNet Model
```
pip3 install efficientnet-pytorch
```
## tqdm progress bar
```
pip3 install tqdm
```

### All documents are equipped with W&B tracking. If you do not want to use W&B, you can comment out the 'wandb' related code.

### If you want to run and test the code, you need to change the path of the file corresponding to the beginning of the code, i.e. DATA_FOLDER_PATH = '[YOUR_REPOSITORIE]/Animals_with_Attributes2/', JPEGIMAGES_FOLDER_PATH = '[YOUR_REPOSITORIE]/Animals_with_Attributes2/JPEGImages/'