# Bengali-Handwritten-Character-Classification
## Authors: Subrato Chakravorty, Aparna Srinivasan

### Description 
We try to identify the three grapheme elements - grapheme root, vowel diacritics, and consonant diacritics from images of handwritten Bengali graphemes, in order to accelerate the digitization of Bengali resources and promote research in Bengali character recognition. The proposed solution is modelled as a multi-task learning problem i.e., classify the three constituent elements in a grapheme - grapheme root, vowel diacritics, and consonant diacritics using a shared deep learning feature extraction module.

The dataset for this project is taken from a Kaggle Competition and can be found [here](https://www.kaggle.com/c/bengaliai-cv19/data). 
![GitHub Logo](/figures/standardarch.png)

Histories of different models can be found in histories folder. Code to train a standard cnn model and a dense net model is present in __src__ folder and code to train other models will be added with time.

### Requirements
All the package requirements can be found in requirements.txt. To install the requirements:
```
git clone https://github.com/SubratoChakravorty/Bengali-Handwritten-Character-Classification
cd Bengali-Handwritten-Character-Classification
pip install -r requirements.txt
```


### To pre-process dataset (Only once)
Download the data from the link given above and then extract all the files and put them under the \input folder.
```
cd src
python create_data.py
```

### To train a model
```
cd src 
python baseline_model.py # baseline (standard cnn) model
python model_densenet.py # dense net model

```


