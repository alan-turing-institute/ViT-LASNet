# Running inference

`test.py` classifies a single TIFF image, or every `.tif`/`.tiff` file in a folder (`-f`), and writes the results to a CSV file (`-o`, default `predictions.csv`). Run `python test/test.py --help` for every option.

The trained weights are on [Hugging Face](https://huggingface.co/Noushineft/ViT-LASNet), with a model card. Pass `-w Noushineft/ViT-LASNet/<sub-folder>` to download them automatically, for example:

```
python test/test.py -m 2 -n 18 -w Noushineft/ViT-LASNet/vit-mislas -f path/to/images -o out.csv
```

You can also pass a local path to the weights, as in the examples below.

The examples below use a small test set from the [UKCEH freshwater plankton project](https://github.com/NERC-CEH/plankton_ml/).

### ViT, 18 classes

Use the `save_pretrained` folder (the one containing `model.safetensors`), not the single-file `.pt` checkpoint.

```
python test/test.py -m 2 -n 18 -w ~/vit_finetuned_MiSLAS_vit_lr5e-05_epochs30/ -o out.csv \
    -f ../plankton_ml/tests/fixtures/test_images/
```

Add `-k 3` to write the top three predictions and their probabilities. `tools/evaluate.py` reads that format.

### ResNet18, 3 classes

```
python test/test.py -m 1 -w ResNet_18_3classes_RGB.pth -o out.csv -f ../plankton_ml/tests/fixtures/test_images/
```

### ResNet18, 18 classes

`-n` sets the number of classes. It is used both to build the model and to name the predictions. Only 3 and 18 are supported.

```
python test/test.py -m 1 -n 18 -w ResNet_18_18classes_RGB.pth -o out.csv -f ../plankton_ml/tests/fixtures/test_images/
```

For a ResNet trained on greyscale images, add `-g`.
