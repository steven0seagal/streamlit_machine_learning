import streamlit as st
from modules.intro.scripts import intro
from modules.DecisionTreeClassifier.scripts import DecisionTreeClassfierFunc

pages = {
    "Intro":intro,
    "DecisionTreeClassifier":DecisionTreeClassfierFunc,
}


demo_name = st.sidebar.selectbox("Choose an algorithm", pages.keys())
pages[demo_name]()