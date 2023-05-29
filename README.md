# Handwritten_MathEq_Solver
The goal is to develop a machine learning model capable of recognizing and interpreting handwritten digits mathematical symbols within an equation image, allowing for the automatic solving of handwritten math problems.

Handwritten Equation Solver is trained by handwritten digits(0-9) and mathematical symbols(+,-,x) using Convolutional Neural Network.
 
Uploaded Image should be with white background and digits/symbols in black.

Apply image processing techniques to the equation image, such as thresholding, noise removal, and contour detection, to extract individual digits and symbols, and use the trained CNN model detect to solve the handwritten equation.

You can run all the three ipynb files either separately or sequentially.
1. For running Data_extraction.ipynb first download train images.rar zip file and extract it in the folder containing Data_extaction.ipynb file.
2. For running model_training.ipynb, you either need to download train_final.csv or you can run it after succesfully running Data_extraction.ipynb.
3. For running CNN_test.ipynb, you either need to download model_final.h5 and model_final.json file or you can run it after succesfully running model_training.ipynb file. You also need to replace the path of the image in code from the local path of image to be tested on your computer.




