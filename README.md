# Handwritten_MathEq_Solver
The goal is to develop a machine learning model capable of recognizing and interpreting handwritten digits mathematical symbols within an equation image, allowing for the automatic solving of handwritten math problems.

Handwritten Equation Solver is trained by handwritten digits(0-9) and mathematical symbols(+,-,x) using Convolutional Neural Network.
 
Uploaded Image should be with white background and digits/symbols in black.

Apply image processing techniques to the equation image, such as thresholding, noise removal, and contour detection, to extract individual digits and symbols, and use the trained CNN model detect to solve the handwritten equation.

You can run all the three ipynb files either separately or sequentially.
1. For running Data_extraction.ipynb first download train images.rar zip file and extract it in the folder containing Data_extaction.ipynb file.
2. For running model_training.ipynb, you either need to download train_final.csv or you can run it after succesfully running Data_extraction.ipynb.
3. For running CNN_test.ipynb, you either need to download model_final.h5 and model_final.json file or you can run it after succesfully running model_training.ipynb file. You also need to replace the path of the image in code from the local path of image to be tested on your computer.
![image](https://github.com/om-sri/Handwritten_MathEq_Solver/assets/81167782/5c3f79ea-10a8-4034-bfbe-975d19f10c7d)
1. Input an image containing a handwritten equation. Convert the image to a binary image and then invert the image(if digits/symbols are in black).
2. We obtain contours of the image by default, it will obtain contours from left to right.
3. Obtain bounding rectangle for each contour.
4. Sometimes, we may get two or more contours for the same digit/symbol. To avoid that, we can check if the bounding rectangle of those two contours overlaps or not. If they overlap, then discard the smaller rectangle.
5. Now, resize all the remaining bounding rectangle to 28 by 28.
6. Using our model, predict the corresponding digit/symbol for each bounding rectangle and store it in a string.
7.After that use ‘eval’ function on the string to solve the equation.

## Deployment 
Streamlit is an open-source framework in Python that helps us in creation and deployment of machine learning web applications.It is a faster way to create and share data apps.

To deploy the application locally:
1. Install Streamlit: **pip install streamlit**.
2. Use any IDE to run the python code.
3. Download the requirements from **requirements.txt** file using **pip install -r requirements.txt**.
4. Create a virtual environment and Run the app using the command: **streamlit run test.py**.

The web app contains a Home page with description about the project and a predict option which leads to a page where the image should be uploaded.Then our model detects the equation and predicts the output of uploaded_file.
Link to the website: https://om-sri-handwritten-matheq-solver-test-b1dffu.streamlit.app/

![image](https://github.com/om-sri/Handwritten_MathEq_Solver/assets/81167782/01eaf6bc-c046-43dd-8e81-529c2b1fef91)







