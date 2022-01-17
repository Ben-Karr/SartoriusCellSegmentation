## Data:
* __src__ | _original competition [data from kaggle](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)_
* __Sartorius__ | _prepared data used for training, can be downloaded [here](https://www.kaggle.com/karrrrrrrr/sartoriuscompact)_
    * __LiveCell__: (_pretrain_)
        * added .png masks
        * change image format to png
        * create lightweight coco format .json annotations (following [this notebook](https://www.kaggle.com/coldfir3/efficient-coco-dataset-generator?scriptVersionId=79100851)). This decreases the file size from ~550MB to 230MB for the training annotations.
    * __SartoriusCellSeg:__ (_competition_)
        * added .png masks
        * copied annotation jsons from [Slawek Biels Dataset](https://www.kaggle.com/slawekbiel/sartorius-cell-instance-segmentation-coco)

## Notebooks:
* __Detectron:__ _Use a custom trainer with 1cycle policy to (pre-)train for the Sartorius instance segmentation challenge_
    * [WandB-Train]([Detectron]-WandB-Train.ipynb) _Record training with Weights & Biases_
    * [CellSeg-Inference]([Detectron]-CellSeg-Inference.ipynb) _Get predictions for test set and create submission; copied from [here](https://www.kaggle.com/slawekbiel/positive-score-with-detectron-3-3-inference)_
* __Switch:__ _Using fastai with timm, switch between classification & segmentation (Unet) while sharing a backbone_
    * [LiveCell-Pretraining]([Switch]-LiveCell-Pretrain.ipynb)
    * [CellSeg-Training]([Switch]-CellSeg-Train.ipynb) 
    * [CellSeg-Inference]([Switch]-CellSeg-Inference.ipynb) _Get predictions and transform from semantic to instance segmentation, create submission_
* __LiveCell:__ _Prepare the livecell data for pretraining_
    * [coco_annotations]([LiveCell]coco_annotations.ipynb)
    * [create_pngMasks]([LiveCell]create_pngMasks.ipynb)
    * [tiff_to_png]([LiveCell]tiff_to_png.ipynb)
* __Sagemaker:__ _Upload big files from sagemaker to dropbox_
    * [UploadToDropbox]([Sagemaker]-UploadToDropbox.ipynb)
