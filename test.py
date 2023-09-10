import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import pickle
from PIL import Image, ImageOps
import cv2
from tensorflow.keras.models import model_from_json

json_file = open('model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_final.h5")


train_data=[]
app_mode = st.sidebar.selectbox('Select Page',['Home','Predict','Feedback'])

if app_mode=='Home':
    st.title('Hand written Math equation solver')
    st.write('The goal is to develop a machine learning model capable of recognizing and interpreting handwritten digits mathematical symbols within an equation image, allowing for the automatic solving of handwritten math problems.')
    st.write('Handwritten Equation Solver is trained by handwritten digits(0-9) and mathematical symbols(+,-,x) using Convolutional Neural Network.')
    st.write('Apply image processing techniques to the equation image, such as thresholding, noise removal, and contour detection, to extract individual digits and symbols, and use the trained CNN model detect to solve the handwritten equation.')
    

elif app_mode == 'Predict':

    st.title('Math Equation Solver')

    st.subheader('Upload Image')
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if st.button('Predict'):
        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert('L')
            img = np.array(image)
         
             # Display the uploaded image
            st.image(img, use_column_width=True)
                
            
            img=~img
            ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            ctrs,ret=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            w=int(28)
            h=int(28)
    
            # print(len(cnt))
            rects=[]
            for c in cnt :
                x,y,w,h= cv2.boundingRect(c)
                rect=[x,y,w,h]
                rects.append(rect)
            # print(rects)
            bool_rect=[]
            for r in rects:
                l=[]
                for rec in rects:
                    flag=0
                    if rec!=r:
                        if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                            flag=1
                        l.append(flag)
                    if rec==r:
                        l.append(0)
                bool_rect.append(l)
            # print(bool_rect)
            dump_rect=[]
            for i in range(0,len(cnt)):
                for j in range(0,len(cnt)):
                    if bool_rect[i][j]==1:
                        area1=rects[i][2]*rects[i][3]
                        area2=rects[j][2]*rects[j][3]
                        if(area1==min(area1,area2)):
                            dump_rect.append(rects[i])
            # print(len(dump_rect)) 
            final_rect=[i for i in rects if i not in dump_rect]
            # print(final_rect)
            for r in final_rect:
                x=r[0]
                y=r[1]
                w=r[2]
                h=r[3]
                im_crop =thresh[y:y+h+10,x:x+w+10]
                im_resize = cv2.resize(im_crop,(28,28))
                im_resize=np.reshape(im_resize,(28,28,1))
                train_data.append(im_resize)
       
    
    
            s=''
            for i in range(len(train_data)):
                train_data[i] = np.array(train_data[i])
                train_data[i] = train_data[i].reshape(1, 28, 28, 1)
                result = np.argmax(loaded_model.predict(train_data[i]), axis=-1)  
                
                if(result[0]==10):
                    s=s+'-'
                if(result[0]==11):
                    s=s+'+'
                if(result[0]==12):
                    s=s+'*'
                if(result[0]==0):
                    s=s+'0'
                if(result[0]==1):
                    s=s+'1'
                if(result[0]==2):
                    s=s+'2'
                if(result[0]==3):
                    s=s+'3'
                if(result[0]==4):
                    s=s+'4'
                if(result[0]==5):
                    s=s+'5'
                if(result[0]==6):
                    s=s+'6'
                if(result[0]==7):
                    s=s+'7'
                if(result[0]==8):
                    s=s+'8'
                if(result[0]==9):
                    s=s+'9'
    
            print(s)    

            st.write('The equation is: ',s)
            result = eval(s)
            st.write('The result is: ',result)

        else:
            st.write('Please upload an Image')

elif app_mode == "Feedback":
    st.title("Feedback and Issue Reporting")

    # Create a text area for users to enter feedback or report issues
    feedback_text = st.text_area("Enter your feedback or report an issue:")

    # Create a button to submit feedback
    if st.button("Submit Feedback"):
        # You can handle the feedback submission here
        # For example, you can send the feedback to an email address or a database
        # You may also want to display a confirmation message
        st.success("Thank you for your feedback! We will review it.")

# Optionally, you can add a link to a support email
st.sidebar.markdown("[Contact Support](mailto:omsrirao14@gmail.com)")
