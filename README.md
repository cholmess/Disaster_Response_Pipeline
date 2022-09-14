# Disaster Response Pipeline Project

### Summary of the project

The project is about creating an API to classify messages in an emergency in order to allocate resources efficiently. Datasets for messages and categories are taken from Appen. A pipeline is created with a MultiOutputClassifier since the output has 36 possible categories.

Instructions are given below to run the ETL and ML pipelines, as well on how to use the Flask app to visualize 

### Files used in the repository

- disaster_messages.csv: dataset of messages sent in an undisclosed emergency with genre and the message sent both in enlish and the original language
- disaster_categories.csv: categories associated with the messages
- DisasterResponse.db: database with cleaned and merged data of previous mentioned files
- classifier.pkl: saved model from the ML pipeline


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage. Front End was made with Flask, model tested with XGBoost


### Conclusion

Model is able to predict with 95% accuracy, 66% precision and 79% recall the categories of the messages in the dataset. It needs improvement, specially on precision, but considering it's a 36 output modell it's a good first approach.  

### Acknowledgement/References

Thanks to Udacity & Bosch for creating the front end with Flask and providing the data 