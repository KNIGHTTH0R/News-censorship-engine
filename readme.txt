Requirements:
1. install tensorflow
2. install flask
3. python3 and other imported modules (numpy, scikitlearn, scipy etc)


-- file train.ipynb
contains our training method and steps.
Make sure you have installed all imported modules.
During training process, our model reached 90.4% of accuracy


--file engine.py
article category predictor based on model developed from train.ipynb file


--file classifier.html
an html file for submitting a will predicted article


--the model folder
a folder containing neural net biases and weight


---- how to run ----
1. run engine.py (python3 engine.py)
2. open classifier.html in web browser, paste your article here, and press "klasifikasi" button.
3. the result will appear (sensitive or nonsensitive)
