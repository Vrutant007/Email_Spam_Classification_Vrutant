import streamlit as st
import pickle

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorization-1.pkl','rb'))

st.title('Email Spam Classification Application')
st.write('This application uses a machine learning model to classify emails as spam or not spam')
user_input = st.text_area('Please enter the Email to classify',height=50)
if st.button('classify'):
    if user_input:
        data = [user_input]
        vect = cv.transform(data).toarray()
        pred = model.predict(vect)
        if pred[0] == 0:
            st.write('This email is not spam')
            st.success('Email is not spam')
        else:
            st.write('This email is spam')
    else:
        print('Please type Email')