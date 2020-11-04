import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    """Salary Predictor with ML"""
    st.title("Salary Data Research")
    activity = ["preview","description","queries","plots","models"]
    choice = st.sidebar.selectbox("Main activity", activity)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                    "class"]
    salary_df = pd.read_csv(url, names=column_names)

    #EDA
    if choice == 'preview':
        st.subheader("Exploratory Data Analysis")
        st.markdown("[Dataset: UCI repository - adult.data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult)")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #Preview
        st.text("Preview")
        if st.checkbox("Preview Dataset"):
            number = st.number_input("Number to Show", format = "%d")
            st.dataframe(salary_df.head(int(number)))
        if st.checkbox("Column Names"):
            st.write(salary_df.columns)
    elif choice == 'description':
        #Description
        st.text("Description")
        if st.checkbox("Show Description"):
            st.write(salary_df.describe())
        #Shape
        if st.checkbox("Show Shape of Dataset"):
            st.write(salary_df.shape)
            data_dim = st.radio("Show dimensions by",("Rows","Columns"))
            if data_dim == "Rows":
                st.text("Number of rows")
                st.write(salary_df.shape[0])
            elif data_dim == "Columns":
                st.text("Number of columns")
                st.write(salary_df.shape[1])
            else:
                st.write(salary_df.shape)
    elif choice == 'queries':
        #Selections
        st.text("Queries")
        if st.checkbox("Select Columns to show"):
            all_columns = salary_df.columns.tolist()
            selected_columns = st.multiselect("Select columns", all_columns)
            new_df = salary_df[selected_columns]
            st.dataframe(new_df)
        if st.checkbox("Select rows to show"):
            select_index = st.multiselect("Select Rows", salary_df.head(10).index)
            selected_rows = salary_df.loc[select_index]
            st.dataframe(selected_rows)
        #Value counts
        if st.button("Value counts"):
            st.text("Value counts by class")
            st.write(salary_df.iloc[:,-1].value_counts())
    elif choice == 'plots':
        #Plot
        st.text("Plots")
        if st.checkbox("Show correlation plot [Matplotlib]"):
            plt.matshow(salary_df.corr())
            st.pyplot()
        if st.checkbox("Show correlation plot [Seaborn]"):
            st.write(sns.heatmap(salary_df.corr(), annot=True))
            st.pyplot()
    elif choice == 'models':
        st.subheader("Be patience, still under construction...")
if __name__ == '__main__':
    main()
