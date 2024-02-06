# Description of Project
-----------------------
Glaucoma is an eye disease that damages the optic nerve and can cause irreversible visual impairment unless diagnosed in its early stages. This disease is the leading cause of blindness in India, affecting 1.2 million individuals and accounts for 5.5%
of the overall cases of total blindness in the country. Early and efficient detection of glaucoma has thus become increasingly paramount, and many methods for such detection have been devised using various techniques such as deep learning.

Existing literature has only utilized either fundus or OCT images for detection which can lead to inaccurate diagnosis. As the fundus image only provides the view of the topmost retinal layer, it may not show structural changes indicating glaucoma during the disease’s early stages as the deeper layers get affected first before the top layers. Furthermore, there is no automated tool available for
glaucoma detection which aids doctors to diagnose the disease quickly on a large scale. Thus, the objective of this research is to present an Enhanced Glaucoma Prediction Mass (EGP-Mass) screening tool for detecting glaucoma accurately from two image modalities, namely fundus and OCT images. This tool can be used in scenarios such as medical eye camps or crowded hospitals, where accuracy and efficiency are the need of the hour.


# URL/Source for Dataset
----------------------

DRISHTI-GS: *https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation*

ACRIMA: *https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135*

Zenodo OCT Images: *https://zenodo.org/record/1481223#.Y20g3XbMIuV*


# Software Requirements:
----------------------

- Tensorflow
- Matplotlib
- Keras
- Flask
- Node JS
- Next JS
- wkhtmltopdf (For Report Generation)


# Hardware Requirements:
-------------------------------
- Systems with good GPUs (Nvidia RTX cards) are recommended for good performance
- Systems with CPU (average performance)
- Google Colaboratory (good performance)


# Source Code Execution Instructions:

## Running the Mass Screening tool:
-------------------------------

1. Navigate to the web-app directory using the command line interface.
2. Open a terminal within the frontend folder of the application.
3. Execute the command "npm install" to install the required dependencies.
4. Enter the command "npm run dev" and execute to initiate the frontend server on the localhost.
5. Open a separate terminal within the backend folder.
6. Execute the command "flask run" to start the backend server, which will run on the localhost.

## Training of Few Shot Learning:
-----------------------------

1. Begin by installing Anaconda and creating a new Conda environment using the command: "conda create -n <env-name> python=3.9".
2. Install all the necessary software requirements specified in the provided file using the command: "pip install <module-name>".
3. Access the FSL-VGG Jupyter notebook.
4. Ensure that the appropriate Conda environment is selected.
5. Open the notebook as a Jupyter Notebook on localhost and meticulously execute each cell consecutively.
6. This sequence of actions will initiate and complete the training of the machine learning model.
7. Once the model is saved, proceed to open the testing Jupyter notebook.
8. Load the previously saved model using its designated name.
9. Execute the notebook cells and record the output metrics for further analysis and evaluation.

## Training of U-Net:
------------------

1. Begin by installing Anaconda and creating a new Conda environment using the following command: "conda create -n <env-name> python=3.9".
2. Subsequently, install all the specified software requirements detailed in the provided file using the command: "pip install <module-name>".
3. Access the "Final_U-net" Jupyter notebook.
4. Ensure the appropriate Conda environment is selected for seamless compatibility.
5. Open the notebook as a Jupyter Notebook on localhost, meticulously executing each cell consecutively.
6. This process will initiate the training of the machine learning model.
7. Upon completion, save the trained model and proceed to load it into the same Jupyter notebook.
8. Execute the notebook cells, recording and preserving the resulting output metrics for analysis.
9. Explore the "test.py" file, which contains the code for calculating the adaptive binarizer.
10. Utilize this code to compute the optimal binarizer for each image as needed.

## Running the pre-trained model:
------------------

1. Initiate the process by installing Anaconda and creating a new Conda environment using the command: "conda create -n <env-name> python=3.9".
2. Proceed to install all the requisite software dependencies outlined in the provided file with the command: "pip install <module-name>".
3. Open the "model.py" file for further configuration and adjustments.
4. Ensure the appropriate Conda environment is selected to guarantee compatibility.
5. Execute the Python script to save the model probabilities into a CSV file, marking the commencement of the model training process.
6. Upon completion of the training, save the trained model.
7. Launch the "analyze_model.py" script for comprehensive model analysis.
8. Execute the file, capturing and preserving the resultant output metrics for thorough examination and documentation.
