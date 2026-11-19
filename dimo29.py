import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="My Wide App",
    layout="wide",  # Options: "centered" (default) or "wide"
    initial_sidebar_state="expanded",
)

st.title("Diamonds Quality Prediction")
##Data Load
df = pd.read_csv("diamonds.csv")
df = df.drop(columns=["Unnamed: 0"])
#st.write(df.head())
#st.table(df.head())
#st.dataframe(df.head())

num_col = ['carat', 'depth', 'table', 'x', 'y', 'z','price']
cat_col = ['cut','color','clarity']

b1 = st.sidebar.checkbox("Show Dataset")
if b1:
    st.write(df.head())

#EDA

col1, col2 = st.columns([1,1])
b2 = st.sidebar.checkbox("EDA")
if b2:
    st.write("EDA")
    s1 = st.sidebar.selectbox("Type of EDA",["Unvariate","Bivariate","Multivariate"])
    if s1 == "Unvariate":
        col_name = st.sidebar.selectbox("Select columns",df.columns)
        if col_name in num_col:
            c1 = px.histogram(df[col_name])
            col1.plotly_chart(c1)

            c2 = px.box(df[col_name])
            col2.plotly_chart(c2)
        elif col_name in cat_col:
            df_cat = df.groupby(col_name)["price"].count().reset_index()
            df_cat.rename(columns={"price":"count"},inplace=True)
            #st.write(df_cat)
            c1 = px.bar(df_cat, x=col_name, y='count')
            col1.plotly_chart(c1)

            c2 = px.pie(df_cat,col_name, values='count')
            col2.plotly_chart(c2)
    elif s1 == "Bivariate":
        col_name = st.sidebar.selectbox("Select columns",df.columns)
        target_col = df['price']

        if col_name in num_col:
            c1 = px.scatter(df, col_name, target_col)
            col1.plotly_chart(c1)

            df_num = df.groupby(col_name)["price"].mean().reset_index()
            c2 = px.line(df_num, x=col_name, y='price')
            col2.plotly_chart(c2)
    else:
        pass

x = df.drop(columns=['price'])
y = df['price']

Test_Size = st.sidebar.slider("Select Test Size",0.10,0.95,0.20,0.05)
Random_state = st.sidebar.number_input("Select Random State",0,100,0)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=Test_Size,random_state=Random_state)

