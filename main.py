import streamlit as st
from modules.intro.scripts import intro
from modules.DecisionTreeClassifier.scripts import DecisionTreeClassfier

pages = {
    "Intro":intro,
    "DecisionTreeClassifier":DecisionTreeClassfier,
}


demo_name = st.sidebar.selectbox("Choose an algorithm", pages.keys())
pages[demo_name]()