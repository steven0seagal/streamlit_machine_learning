import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
def DecisionTreeClassfier():

    tab1,tab2,tab3 = st.tabs(['🗃 Raw data','Prepare and evaluate model', 'Make prediction' ])

    with tab1:
        data = pd.read_csv('modules/DecisionTreeClassifier/data/vgsales.csv')
        st.dataframe(data)
        st.dataframe(data.describe())
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
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    with tab3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)