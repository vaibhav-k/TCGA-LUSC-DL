# TCGA-LUSC prognosis prediction

This project contains the implementation to of 3 (three) Deep Learning models, Multi-layered Perceptron (MLP), Convolutional Neural Network (CNN), and Long Shirt Term Memory (LSTM), that were trained on the clinical data of 503 (training + testing) patients (samples).  

Each file in the list below has its use along with how to run it explained:  

- clean_df.: This file contains the code for cleaning and filtering the original TCGA-LUSC dataset to only retain the valid/non-NA values. To run it, simply type `python3 clean_df.py` into the terminal.  
- cnn.py: This file contains the code for training and evaluating both the vanilla CNN and the CNN with transfer learning (VGG16). To run it, simply type `python3 cnn.py` into the terminal.  
- dataset_exploration.py: This file contains the code to explore the TCGA-LUSC clinical dataset. To run it, simply type `python3 dataset_exploration.py` into the terminal.  
- feature_importance.py: This file contains the code to run Boruta's analysis with the cleaned TCGA-LUSC dataset. To run it, simply type `python3 feature_importance.py` into the terminal.  
- fix_data.py: This file contains the code to fix the data and make it fito for feeding to the neural networks. Things like converting the categorical variables to numerical and one-hot encoding the target variable are implemented in this file. To run it, simply type `python3 fix_data.py` into the terminal.  
- lr.py: This file contains the code for training and evaluating a logistic regression model on the TCGA-LUSC dataset. To run it, simply type python3 lr.py into the terminal.  
- lstm.py: This file contains the code for training and evaluating an LSTM model. To run it, simply type `python3 lstm.py` into the terminal.  
- mlp.py: This file contains the code for training and evaluating an MLP model. To run it, simply type `python3 mlp.py` into the terminal.  
- plot_acc.py: This file contains the code to create a matplotlib scatter plot of all the model's accuracies on the TCGA-LUSC clinical dataset. It also plots a horizontal red line to mark the baseline. To run it, simply type `python3 plot_acc.py` into the terminal.  

