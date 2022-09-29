# Experiments for "Expressive Monotonic Networks"

## Environment
First, create a conda environment with the `requirements.yml`.  
Sometimes conda environments might be a pain.  
What you need is pytorch(1.12), numpy, tqdm, torchvision(0.13), pandas and scikit-learn.  
Earlier torch versions also work, but torchvision needs to be at least 0.12, since ResNet weights have a different API in earlier versions.  


## Running experiments
We provide the data for BlogFeedback, COMPAS and LoanDefaulter datasets in the `data` folder.  
Simply running `blogfeedback[_mini].py`, `compas.py` and `loan[_mini].py` will train the models and print out results.  

For the chest xray experiment, the dataset has to be downloaded and unzipped:  
https://www.kaggle.com/datasets/nih-chest-xrays/sample

The path to which you unzip the dataset should be specified in `chest_config.py`

Then you can run, in order:
1. `chest_preprocessing.py`: this will preprocess the data and save torch tensors in the path you specified  
2. `chest_apply_resnet.py`: this will apply the resnet18 model with pretrained weights to the data and save the results in the path you specified  
3. `chest_classify.py`: train the monotonic classifier on the resnet18 features and tabular data and print out results
4. `chest_classify_finetuning.py`: take the classifier trained in `chest_classify.py` and the resnet model and train both end to end again.

We include the pretrained `chest_classify.py` model in the `models` folder.
