# CNN based wrist-specific predictive model

Python code repository for "A Deep Learning Method to Predict Energy Expenditure and Activity Intensity in Free Living Conditions using Wrist-specific Accelerometry".

### Required Modules
* Python 3
* Tensorflow
* Keras
* matplotlib
* numpy
* seaborn
* pandas
* tqdm
* Scikit-learn

### Getting started  

A sample dataset is contained in the data/train_ready folder as numpy data objects.  
  
The model architecture can be found in `modeling/model_train_class.py` and `modeling/model_train_reg.py` files.
To train the model using the sample data, run the `modeling/model_train_both.py` file.  

The pre-trained models developed using the free-living experiment discussed in the research article can be found in `pre-trained-models/` folder.

