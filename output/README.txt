Instructions to run the evaluation script:

cd output
run 'python evaluate.py' 

The model file and the vocabulary (pickle file) gets downloaded from google drive
and the program generates 'test.out' which contain the model predictions on 'test.csv'
(1000 test samples). The score at end of the run is the accuracy of the model on these test samples based on the references (reference.out).

The index of data points used in the evaluation/analysis of results in the report are:
4 - Sample 1
132 - Sample 2
710 - Sample 3
168 - Sample 4
322 - Sample 5
329 - Sample 6