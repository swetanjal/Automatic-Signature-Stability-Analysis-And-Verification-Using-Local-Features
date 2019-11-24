# Automatic Signature Stability Analysis And Verification Using Local Features

## Installation

Clone the repository [here](https://github.com/swetanjal/Automatic-Signature-Stability-Analysis-And-Verification-Using-Local-Features).

```bash
   >>> git clone https://github.com/swetanjal/Automatic-Signature-Stability-Analysis-And-Verification-Using-Local-Features
```
## Usage

Run Demo (Run Train model before hand)
```bash
   >>> python3 demo.py
```

Train model
```bash
   >>> python3 main_train.py
```


Test model
```bash
   >>> python3 main_test.py
```

## Requisites
 - `opencv-contrib-python:` Contains implementation of SURF
 ```bash
    >>> pip3 install opencv-contrib-python
 ```
 - `threading:` This code uses threading to parallelize

## Folders
 - `Documents:` Research Paper & Project Proposal
 - `Pickles:` Pickle files containing reference database and other intermediate values
 - `Plots:` Contains all generated plots
 - `src:` Code Folder
 - `TestSet:` Test Set of 4NSigComp2010 Dataset
 - `TrainSet:` Train Set of 4NSigComp2010 Dataset

## Code Files
 - `classify.py:` Classify matched keypoints
 - `createDB.py:` Create the reference database
 - `demo.py:` Demo for the presentation (Incomplete)
 - `main_train.py:` Generate graphs and results for train set
 - `main_test.py:` Generate graphs and results for test set
 - `plots.py:` Plotting functions

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
[Nishant Sharma](https://github.com/nishanth2358), [Swetanjal Dutta](https://github.com/swetanjal), [Teja Sai Dhondu](https://github.com/TD87)
Please make sure to update tests as appropriate.

## Dataset
Dataset and instructions to download can be found [here](http://www.iapr-tc11.org/mediawiki/index.php/ICFHR_2010_Signature_Verification_Competition_(4NSigComp2010)). TrainingSet and TestSet folders must be place in the repository's root folder


## License
[MIT](https://choosealicense.com/licenses/mit/)
