# Disaster

This project seeks to classify tweets surrounding natural disasters into one of 36 categories. It also includes a web app that allows the user to classify new tweets into the existing categories. 

Though there are many files in this repo only three are needed to run the app, the data preparation file, the model building file and the app running file itself. 

First, run the data pipeline, 

Second, run the model, 'train_classifier.py', which will create the model and pickle it in a file called 'classifier.pkl'. In the command line from inside the 'Disaster' folder run this code: 

python model/train_classifier.py sqlite:///data/DisasterResponse.db classifier.pkl'

Third, and finally, run the app from your command line. From inside the 'Disaster' folder run the code:

python app/run.py

The directions will be returned to the command line but basically all you have to do is open a browser and point it to http://0.0.0.0:3001/ .

