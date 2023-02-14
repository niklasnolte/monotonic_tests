# Experiments for "Expressive Monotonic Networks"

Dependencies: torch, torchvision, numpy, pandas, tqdm, sklearn, monotonenorm (pip)

We provide the data for BlogFeedback, COMPAS and LoanDefaulter datasets in the `data` folder.
Simply running `blogfeedback[_mini].py`, `compas.py`, `loan[_mini].py`, `heart_disease.py` and `auto_mpg.py` will train the models and print out results.

For the chest xray experiment, the dataset has to be downloaded and unzipped:
https://www.kaggle.com/datasets/nih-chest-xrays/sample

The path to which you unzip the dataset should be specified in `chest_config.py`

Then you can run, in order:
1. `chest_preprocessing.py`: this will preprocess the data and save torch tensors in the path you specified
2. `chest_apply_resnet.py`: this will apply the resnet18 model with pretrained weights to the data and save the results in the path you specified
3. `chest_classify.py`: train the monotonic classifier on the resnet18 features and tabular data and print out results
4. `chest_classify_finetuning.py`: take the classifier trained in `chest_classify.py` and the resnet model and train both end to end again.

We include the pretrained `chest_classify.py` model in the `models` folder.
