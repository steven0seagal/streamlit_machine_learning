import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree
import graphviz
def DecisionTreeClassfierFunc():

    st.write('The goal of this problem is to predict whether the balance scale will tilt to the left or right based '
             'on the weights on the two sides and distance from middle point.')


    tab1,tab2,tab3,tab4 = st.tabs(['🗃 Raw data','Prepare and evaluate model', 'Make prediction','Model chart' ])

    with tab1:
        data = pd.read_csv('modules/DecisionTreeClassifier/data/balance-scale.data',delimiter=',')
        st.dataframe(data)
        st.dataframe(data.describe())
        # st.write(data.info())

        button = st.button('Create data raport')
        if button:
            profile = ProfileReport(data)
            file_name = "modules/DecisionTreeClassifier/data/output.html"
            profile.to_file(file_name)

            with open(file_name, "rb") as file:
                st.download_button(
                    label="Download raport",
                    data=file,
                    file_name='raport.html',
                    mime='html',
                )
    with tab2:
        button = st.button('Click to create model')
        if button:

            X = data.drop(columns=['ClassName'])
            y = data['ClassName']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            model = DecisionTreeClassifier()
            model.fit(X_train,y_train)
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            st.write(f'Accuracy of created model is {score}, if result is acceptable please save it ')
            st.write('Now you can save model and go to last tab to make predictions')
            # save_button = st.button(label='Save model for future use')
            joblib.dump(model,'modules/DecisionTreeClassifier/data/model/tilt_recommender.joblib')
            with open('modules/DecisionTreeClassifier/data/model/model_info','w') as handler:
                handler.write(str(score))


    with tab3:
        try:
            with open('modules/DecisionTreeClassifier/data/model/model_info', 'r') as handler:
                score = handler.read().strip()
            model = joblib.load('modules/DecisionTreeClassifier/data/model/tilt_recommender.joblib')
            st.write(f'Current model has {score} accuracy')
            with st.form("inser_data"):
                st.write("Write values to predict balance")
                left_weigth = st.number_input(label='Weigth on left side', min_value=0)
                right_weigth = st.number_input(label='Weigth on right side', min_value=0)
                left_distance =st.number_input(label='Distance from middle to left', min_value=0)
                right_distance =st.number_input(label='Distance from middle to right', min_value=0)
                prediction_one = model.predict([[left_weigth,left_distance,right_weigth,right_distance]])
                submitted = st.form_submit_button("Submit")
                if submitted:

                    st.write(f"Prediction is {prediction_one[0]}")
        except FileNotFoundError:
            st.write('There is no model to use')

    with tab4:
        #TODO: add tree making

        try:

            model = joblib.load('modules/DecisionTreeClassifier/data/model/tilt_recommender.joblib')
            tree.export_graphviz(model,out_file='modules/DecisionTreeClassifier/data/model/tilt_recommender.tree')
        except FileNotFoundError:
            st.write('There is no model to use')
