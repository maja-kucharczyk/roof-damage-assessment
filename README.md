# Delineate post-storm roof damage using deep learning and aerial imagery

<img src="https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/0_RoofDecking.jpg?raw=true" width="50%"><img src="https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/0_RoofHole.jpg?raw=true" width="50%">

Roof damage caused by hurricanes and other storms needs to be rapidly identified and repaired to help communities recover from catastrophic events and support the well-being of residents. Traditional, ground-based inspections are time-consuming but have recently been expedited via manual interpretation of remote sensing imagery. To potentially accelerate the process, automated methods involving artificial intelligence (i.e., [deep learning](https://pro.arcgis.com/en/pro-app/latest/help/analysis/deep-learning/what-is-deep-learning-.htm)) can be applied. 

In this tutorial, you will perform a workflow for training and evaluating deep learning image segmentation models that detect and delineate two classes of post-storm roof damage: roof decking and roof holes. This guide supports the reproducibility of data and results presented in the following publication:

Kucharczyk, M., Nesbit, P. R., & Hugenholtz, C. H. (2025). Automated Mapping of Post-Storm Roof Damage Using Deep Learning and Aerial Imagery: A Case Study in the Caribbean. *Remote Sensing*, *17*(20), 3456. https://doi.org/10.3390/rs17203456

> [!NOTE]
> This tutorial was last tested on December 27, 2025, using ArcGIS Pro 3.4.3 and Jupyter Notebook 7.2.1. If you're using different versions, you may encounter different functionality and results.

## Requirements
- Esri ArcGIS Pro with Image Analyst extension
- [Deep learning packages for ArcGIS Pro](https://pro.arcgis.com/en/pro-app/latest/help/analysis/deep-learning/install-deep-learning-frameworks.htm)
- Recommended: NVIDIA GPU with a minimum of 8 GB of dedicated memory
- For more guidance, visit the [Get ready for deep learning in ArcGIS Pro](https://learn.arcgis.com/en/projects/get-ready-for-deep-learning-in-arcgis-pro/) tutorial and [Deep learning frequently asked questions](https://pro.arcgis.com/en/pro-app/latest/help/analysis/deep-learning/deep-learning-faq.htm).

## Workflow overview
This tutorial has five major steps:
1. [Prepare images](#1.-Prepare-images)
2. [Export training data](#2.-Export-training-data)
3. [Train deep learning models](#3.-Train-deep-learning-models)
4. [Delineate roof damage](#4.-Delineate-roof-damage)
5. [Evaluate deep learning models](#5.-Evaluate-deep-learning-models)

> [!NOTE]
> You do not need to complete each step. There are instructions for downloading the required files at the beginning of each step.

These steps are part of a comprehensive deep learning workflow, as shown below. To support the reproducibility of the data and results presented in the [publication](https://doi.org/10.3390/rs17203456), this tutorial does not cover the creation of image boundary polygons, training polygons, and reference polygons. Instead, you will download these polygons.

![](https://raw.githubusercontent.com/maja-kucharczyk/roof-damage-assessment/bf2f721916c8a907b46078acb9f2100c3f1ad4fd/img/0_Workflow.svg)

---

## 1. Prepare images

### Download files
1. Go to the [file repository](https://drive.google.com/drive/folders/11AbOA5j1tVF8iIRS7vwDq8YJcy1Z3hhy).
2. Download and unzip the following files:
    - Tools.zip
    - Polygons.zip
    - Downloaded_Test_Images.zip
    - Downloaded_Training_Images_Dominica.zip
    - Downloaded_Training_Images_SintMaarten.zip
    - Downloaded_Training_Images_TheBahamas.zip
    - Downloaded_Training_Images_USVI.zip

### Prepare Dominica training images
1. Open ArcGIS Pro and start a new project.
2. In the Catalog pane, right-click **Folders** and select **Add Folder Connection**.
3. In the new window, select the folder that contains the downloaded files, and select **OK** to add the folder connection.
4. In the Catalog pane, right-click the folder that contains the downloaded files and select **Make Default**.
5. In the Catalog Pane, expand the folder, expand **Tools**, expand **Roof Damage Assessment.atbx**, and double-click **Prepare Images**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/1_Dominca_5.JPG?raw=true)

6. In the Prepare Images tool, for *Downloaded Images Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Downloaded_Training_Images_Dominica**, and select **OK**.
7. In the Prepare Images tool, for *Image Boundary Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Image_Boundary_Polygons.gdb**, and select **OK**.
8. In the Prepare Images tool, for *Output Prepared Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Prepared_Training_Images_Dominica*, press **Enter**, select the new file geodatabase, and select **OK**.
9. In the Prepare Images tool, for *Scratch File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Scratch*, press **Enter**, select the new file geodatabase, and select **OK**.
10. In the Prepare Images tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/1_Dominica_10.JPG?raw=true)

11. While the tool is running, the progress bar indicates the current preparation step and image. To track which images have been prepared and how many are left, select **View Details**.
12. Once the tool is done running, all input downloaded images have been prepared.

### Prepare Sint Maarten training images
1. With the Prepare Images tool still open, for *Downloaded Images Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Downloaded_Training_Images_SintMaarten**, and select **OK**.
2. In the Prepare Images tool, for *Output Prepared Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Prepared_Training_Images_SintMaarten*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Prepare Images tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/1_SintMaarten_3.JPG?raw=true)

4. Once the tool is done running, all input downloaded images have been prepared.

### Prepare The Bahamas training images
1. With the Prepare Images tool still open, for *Downloaded Images Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Downloaded_Training_Images_TheBahamas**, and select **OK**.
2. In the Prepare Images tool, for *Output Prepared Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Prepared_Training_Images_TheBahamas*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Prepare Images tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/1_TheBahamas_3.JPG?raw=true)

4. Once the tool is done running, all input downloaded images have been prepared.

### Prepare US Virgin Islands training images
1. With the Prepare Images tool still open, for *Downloaded Images Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Downloaded_Training_Images_USVI**, and select **OK**.
2. In the Prepare Images tool, for *Output Prepared Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Prepared_Training_Images_USVI*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Prepare Images tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/1_USVI_3.JPG?raw=true)

4. Once the tool is done running, all input downloaded images have been prepared.

### Prepare test images
1. With the Prepare Images tool still open, for *Downloaded Images Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Downloaded_Test_Images**, and select **OK**.
2. In the Prepare Images tool, for *Output Prepared Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Prepared_Test_Images*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Prepare Images tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/1_TestImages_3.JPG?raw=true)

4. Once the tool is done running, all input downloaded images have been prepared.

[Back to top](#Delineate-post-storm-roof-damage-using-deep-learning-and-aerial-imagery)

---

## 2. Export training data

### Download files
1. Go to the [file repository](https://drive.google.com/drive/folders/11AbOA5j1tVF8iIRS7vwDq8YJcy1Z3hhy).
2. Download and unzip the following files if they were not already downloaded or created in a previous step:
    - Tools.zip
    - Polygons.zip
    - Prepared_Training_Images_Dominica.gdb.zip
    - Prepared_Training_Images_SintMaarten.gdb.zip
    - Prepared_Training_Images_TheBahamas.gdb.zip
    - Prepared_Training_Images_USVI.gdb.zip

### Export dual-class training data (Dominica and Sint Maarten)
1. Perform Steps 1-4 of *[Prepare Dominica training images](#Prepare-Dominica-training-images)* if you have not yet created a new ArcGIS Pro project, connected to the downloaded files folder, and set the folder as the default folder.
2. In the Catalog pane, expand the folder that contains the downloaded files, expand **Tools**, expand **Roof Damage Assessment.atbx**, and double-click **Export Training Data**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_DualClass_2.JPG?raw=true)

3. In the Export Training Data tool, for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_Dominica.gdb**, and select **OK**.
4. In the Export Training Data tool, for *Training Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Training_Polygons_RoofDecking_RoofHole.gdb**, and select **OK**.
5. In the Export Training Data tool, for *Image Boundary Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Image_Boundary_Polygons.gdb**, and select **OK**.
6. In the Export Training Data tool, for *Output Training Data Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **Folder**, type *Training_Dataset_Dominica_SintMaarten_RoofDecking_RoofHole*, press **Enter**, select the new folder, and select **OK**.
7. In the Export Training Data tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_DualClass_7.JPG?raw=true)

8. While the tool is running, the progress bar indicates the current image from which training data are being exported. To track which images have been used and how many are left, select **View Details**.
9. Once the tool is done running, for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_SintMaarten.gdb**, and select **OK**.
10. In the Export Training Data tool, select **Run**. This will append the dataset with training data from Sint Maarten.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_DualClass_10.JPG?raw=true)

11. Once the tool is done running, the training dataset is complete.

### Export roof decking training data (Dominica and Sint Maarten)
1. With the Export Training Data tool still open, for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_Dominica.gdb**, and select **OK**.
2. In the Export Training Data tool, for *Training Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Training_Polygons_RoofDecking.gdb**, and select **OK**.
3. In the Export Training Data tool, for *Output Training Data Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **Folder**, type *Training_Dataset_Dominica_SintMaarten_RoofDecking*, press **Enter**, select the new folder, and select **OK**.
4. In the Export Training Data tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_RoofDecking_4.JPG?raw=true)

5. Once the tool is done running, for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_SintMaarten.gdb**, and select **OK**.
6. In the Export Training Data tool, select **Run**. This will append the dataset with training data from Sint Maarten.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_RoofDecking_6.JPG?raw=true)

7. Once the tool is done running, the training dataset is complete.

### Export expanded roof decking training data (The Bahamas and US Virgin Islands)
1. To export an expanded roof decking training dataset with data from the Bahamas and US Virgin Islands, the roof decking training dataset must first be duplicated. In the Catalog pane, right-click the folder that contains the downloaded files and select **Show In File Explorer**. In the File Explorer window, copy and paste **Training_Dataset_Dominica_SintMaarten_RoofDecking**. Once pasted, rename the folder *Training_Dataset_Dominica_SintMaarten_TheBahamas_USVI_RoofDecking*.
2. In the Export Training Data tool (Geoprocessing pane), for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_TheBahamas.gdb**, and select **OK**.
3. In the Export Training Data tool, for *Output Training Data Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Refresh**, select **Training_Dataset_Dominica_SintMaarten_TheBahamas_USVI_RoofDecking**, and select **OK**.
4. In the Export Training Data tool, select **Run**. This will append the dataset with training data from the Bahamas.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofDecking_4.JPG?raw=true)

5. Once the tool is done running, for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_USVI.gdb**, and select **OK**.
6. In the Export Training Data tool, select **Run**. This will append the dataset with training data from US Virgin Islands.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofDecking_6.JPG?raw=true)

7. If the tool fails, select **View Details** to view the error message. In the new window, scroll to the bottom of the error message. If the message states, *ERROR 002860: Tool parameters are inconsistent with the data you are trying to append to*, close the window and perform the next steps.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofDecking_7a.JPG?raw=true)
![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofDecking_7b.JPG?raw=true)

8. In the Catalog pane, expand the folder that contains the downloaded files, right-click **Training_Dataset_Dominica_SintMaarten_TheBahamas_USVI_RoofDecking**, and select **Show In File Explorer**.
9. In the File Explorer window, right-click **esri_accumulated_stats** and select **Edit**.
10. In the text editor window, in lines 70, 71, and 72, replace *""* with *"Band_1"*, *"Band_2"*, and *"Band_3"*, respectively. Save and close the file.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofDecking_10.JPG?raw=true)

11. In the Export Training Data tool (Geoprocessing pane), select **Run**. This will append the dataset with training data from US Virgin Islands.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofDecking_6.JPG?raw=true)

12. Once the tool is done running, the training dataset is complete.

### Export roof hole training data (Dominica and Sint Maarten)
1. With the Export Training Data tool still open, for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_Dominica.gdb**, and select **OK**.
2. In the Export Training Data tool, for *Training Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Training_Polygons_RoofHole.gdb**, and select **OK**.
3. In the Export Training Data tool, for *Output Training Data Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **Folder**, type *Training_Dataset_Dominica_SintMaarten_RoofHole*, press **Enter**, select the new folder, and select **OK**.
4. In the Export Training Data tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_RoofHole_4.JPG?raw=true)

5. Once the tool is done running, for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_SintMaarten.gdb**, and select **OK**.
6. In the Export Training Data tool, select **Run**. This will append the dataset with training data from Sint Maarten.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_RoofHole_6.JPG?raw=true)

7. Once the tool is done running, the training dataset is complete.

### Export expanded roof hole training data (The Bahamas and US Virgin Islands)
1. To export an expanded roof hole training dataset with data from the Bahamas and US Virgin Islands, the roof hole training dataset must first be duplicated. In the Catalog pane, right-click the folder that contains the downloaded files and select **Show In File Explorer**. In the File Explorer window, copy and paste **Training_Dataset_Dominica_SintMaarten_RoofHole**. Once pasted, rename the folder *Training_Dataset_Dominica_SintMaarten_TheBahamas_USVI_RoofHole*.
2. In the Export Training Data tool (Geoprocessing pane), for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_TheBahamas.gdb**, and select **OK**.
3. In the Export Training Data tool, for *Output Training Data Folder*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Refresh**, select **Training_Dataset_Dominica_SintMaarten_TheBahamas_USVI_RoofHole**, and select **OK**.
4. In the Export Training Data tool, select **Run**. This will append the dataset with training data from the Bahamas.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofHole_4.JPG?raw=true)

5. Once the tool is done running, for *Prepared Training Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Training_Images_USVI.gdb**, and select **OK**.
6. In the Export Training Data tool, select **Run**. This will append the dataset with training data from US Virgin Islands.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofHole_6.JPG?raw=true)

7. If the tool fails, select **View Details** to view the error message. In the new window, scroll to the bottom of the error message. If the message states, *ERROR 002860: Tool parameters are inconsistent with the data you are trying to append to*, close the window and perform the next steps.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofHole_7a.JPG?raw=true)
![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofHole_7b.JPG?raw=true)

8. In the Catalog pane, expand the folder that contains the downloaded files, right-click **Training_Dataset_Dominica_SintMaarten_TheBahamas_USVI_RoofHole**, and select **Show In File Explorer**.
9. In the File Explorer window, right-click **esri_accumulated_stats** and select **Edit**.
10. In the text editor window, in lines 70, 71, and 72, replace *""* with *"Band_1"*, *"Band_2"*, and *"Band_3"*, respectively. Save and close the file.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofHole_10.JPG?raw=true)

11. In the Export Training Data tool (Geoprocessing pane), select **Run**. This will append the dataset with training data from US Virgin Islands.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/2_ExpandedRoofHole_6.JPG?raw=true)

12. Once the tool is done running, the training dataset is complete.

[Back to top](#Delineate-post-storm-roof-damage-using-deep-learning-and-aerial-imagery)

---

## 3. Train deep learning models

### Download files
1. Go to the [file repository](https://drive.google.com/drive/folders/11AbOA5j1tVF8iIRS7vwDq8YJcy1Z3hhy).
2. Download and unzip the following files if they were not already downloaded or created in a previous step:
    - Tools.zip
    - Training_Dataset_Dominica_SintMaarten_RoofDecking_RoofHole.zip
    - Training_Dataset_Dominica_SintMaarten_RoofDecking.zip
    - Training_Dataset_Dominica_SintMaarten_RoofHole.zip
    - Training_Dataset_Dominica_SintMaarten_TheBahamas_USVI_RoofDecking.zip
    - Training_Dataset_Dominica_SintMaarten_TheBahamas_USVI_RoofHole.zip

### Train dual-class model (Dominica and Sint Maarten training data)
1. In File Explorer, navigate to the downloaded files folder > **Tools**. Create a copy of **Train Dual-Class Model**. Rename the copy *Dominica_SintMaarten_RoofDecking_RoofHole*.
2. Start Jupyter Notebook*.
    - If the downloaded files folder is on the C drive, select **Start** > **ArcGIS** > **Jupyter Notebook**.
    - If the downloaded files folder is on another drive, select **Start** > **ArcGIS** > **Python Command Prompt**. Then, type ```jupyter notebook --notebook-dir=D:\``` (replacing ```D``` with the correct letter) and press **Enter**.
3. Once Jupyter Notebook launches, a file directory is shown. Double-click through the directory and navigate to the downloaded files folder > **Tools**. Double-click **Dominica_SintMaarten_RoofDecking_RoofHole.ipynb** to open the notebook.
4. In the notebook, click inside the second code cell and replace ```insert_path_here``` with the path to the training data folder.
5. In the notebook, click inside the third code cell and replace ```insert_path_here``` with the path to the output folder where the trained model will be saved.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/3_DualClass_5.JPG?raw=true)

6. Save the notebook (**File** > **Save Notebook**).
7. Run all notebook cells (**Run** > **Run All Cells**).
8. Once all cells are done running, save the notebook (**File** > **Save Notebook**), exit the notebook (**File** > **Close and Shut Down Notebook** > **Ok**), and exit Jupyter Notebook (**File** > **Log Out**).

> [!NOTE]
> The notebook can also be opened and run in ArcGIS Pro.

### Train roof decking model (Dominica and Sint Maarten training data)
1. In File Explorer, navigate to the downloaded files folder > **Tools**. Create a copy of **Train Single-Class Model**. Rename the copy *Dominica_SintMaarten_RoofDecking*.
2. Start Jupyter Notebook*.
    - If the downloaded files folder is on the C drive, select **Start** > **ArcGIS** > **Jupyter Notebook**.
    - If the downloaded files folder is on another drive, select **Start** > **ArcGIS** > **Python Command Prompt**. Then, type ```jupyter notebook --notebook-dir=D:\``` (replacing ```D``` with the correct letter) and press **Enter**.
3. Once Jupyter Notebook launches, a file directory is shown. Double-click through the directory and navigate to the downloaded files folder > **Tools**. Double-click **Dominica_SintMaarten_RoofDecking.ipynb** to open the notebook.
4. In the notebook, click inside the second code cell and replace ```insert_path_here``` with the path to the training data folder.
5. In the notebook, click inside the third code cell and replace ```insert_path_here``` with the path to the output folder where the trained model will be saved.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/3_RoofDecking_5.JPG?raw=true)

6. Save the notebook (**File** > **Save Notebook**).
7. Run all notebook cells (**Run** > **Run All Cells**).
8. Once all cells are done running, save the notebook (**File** > **Save Notebook**), exit the notebook (**File** > **Close and Shut Down Notebook** > **Ok**), and exit Jupyter Notebook (**File** > **Log Out**).

> [!NOTE]
> The notebook can also be opened and run in ArcGIS Pro.

### Train roof decking model (Dominica, Sint Maarten, The Bahamas, and US Virgin Islands training data)
1. In File Explorer, navigate to the downloaded files folder > **Tools**. Create a copy of **Train Single-Class Model**. Rename the copy *Dominica_SintMaarten_TheBahamas_USVI_RoofDecking*.
2. Start Jupyter Notebook*.
    - If the downloaded files folder is on the C drive, select **Start** > **ArcGIS** > **Jupyter Notebook**.
    - If the downloaded files folder is on another drive, select **Start** > **ArcGIS** > **Python Command Prompt**. Then, type ```jupyter notebook --notebook-dir=D:\``` (replacing ```D``` with the correct letter) and press **Enter**.
3. Once Jupyter Notebook launches, a file directory is shown. Double-click through the directory and navigate to the downloaded files folder > **Tools**. Double-click **Dominica_SintMaarten_TheBahamas_USVI_RoofDecking.ipynb** to open the notebook.
4. In the notebook, click inside the second code cell and replace ```insert_path_here``` with the path to the training data folder.
5. In the notebook, click inside the third code cell and replace ```insert_path_here``` with the path to the output folder where the trained model will be saved.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/3_ExpandedRoofDecking_5.JPG?raw=true)

6. Save the notebook (**File** > **Save Notebook**).
7. Run all notebook cells (**Run** > **Run All Cells**).
8. Once all cells are done running, save the notebook (**File** > **Save Notebook**), exit the notebook (**File** > **Close and Shut Down Notebook** > **Ok**), and exit Jupyter Notebook (**File** > **Log Out**).

> [!NOTE]
> The notebook can also be opened and run in ArcGIS Pro.

### Train roof hole model (Dominica and Sint Maarten training data)
1. In File Explorer, navigate to the downloaded files folder > **Tools**. Create a copy of **Train Single-Class Model**. Rename the copy *Dominica_SintMaarten_RoofHole*.
2. Start Jupyter Notebook*.
    - If the downloaded files folder is on the C drive, select **Start** > **ArcGIS** > **Jupyter Notebook**.
    - If the downloaded files folder is on another drive, select **Start** > **ArcGIS** > **Python Command Prompt**. Then, type ```jupyter notebook --notebook-dir=D:\``` (replacing ```D``` with the correct letter) and press **Enter**.
3. Once Jupyter Notebook launches, a file directory is shown. Double-click through the directory and navigate to the downloaded files folder > **Tools**. Double-click **Dominica_SintMaarten_RoofHole.ipynb** to open the notebook.
4. In the notebook, click inside the second code cell and replace ```insert_path_here``` with the path to the training data folder.
5. In the notebook, click inside the third code cell and replace ```insert_path_here``` with the path to the output folder where the trained model will be saved.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/3_RoofHole_5.JPG?raw=true)

6. Save the notebook (**File** > **Save Notebook**).
7. Run all notebook cells (**Run** > **Run All Cells**).
8. Once all cells are done running, save the notebook (**File** > **Save Notebook**), exit the notebook (**File** > **Close and Shut Down Notebook** > **Ok**), and exit Jupyter Notebook (**File** > **Log Out**).

> [!NOTE]
> The notebook can also be opened and run in ArcGIS Pro.

### Train roof hole model (Dominica, Sint Maarten, The Bahamas, and US Virgin Islands training data)
1. In File Explorer, navigate to the downloaded files folder > **Tools**. Create a copy of **Train Single-Class Model**. Rename the copy *Dominica_SintMaarten_TheBahamas_USVI_RoofHole*.
2. Start Jupyter Notebook*.
    - If the downloaded files folder is on the C drive, select **Start** > **ArcGIS** > **Jupyter Notebook**.
    - If the downloaded files folder is on another drive, select **Start** > **ArcGIS** > **Python Command Prompt**. Then, type ```jupyter notebook --notebook-dir=D:\``` (replacing ```D``` with the correct letter) and press **Enter**.
3. Once Jupyter Notebook launches, a file directory is shown. Double-click through the directory and navigate to the downloaded files folder > **Tools**. Double-click **Dominica_SintMaarten_TheBahamas_USVI_RoofHole.ipynb** to open the notebook.
4. In the notebook, click inside the second code cell and replace ```insert_path_here``` with the path to the training data folder.
5. In the notebook, click inside the third code cell and replace ```insert_path_here``` with the path to the output folder where the trained model will be saved.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/3_ExpandedRoofHole_5.JPG?raw=true)

6. Save the notebook (**File** > **Save Notebook**).
7. Run all notebook cells (**Run** > **Run All Cells**).
8. Once all cells are done running, save the notebook (**File** > **Save Notebook**), exit the notebook (**File** > **Close and Shut Down Notebook** > **Ok**), and exit Jupyter Notebook (**File** > **Log Out**).

> [!NOTE]
> The notebook can also be opened and run in ArcGIS Pro.

[Back to top](#Delineate-post-storm-roof-damage-using-deep-learning-and-aerial-imagery)

---

## 4. Delineate roof damage

### Download files
1. Go to the [file repository](https://drive.google.com/drive/folders/11AbOA5j1tVF8iIRS7vwDq8YJcy1Z3hhy).
2. Download and unzip the following files if they were not already downloaded or created in a previous step:
    - Tools.zip
    - Prepared_Test_Images.gdb.zip
    - Dominica_SintMaarten_RoofDecking_RoofHole.zip
    - Dominica_SintMaarten_RoofDecking.zip
    - Dominica_SintMaarten_RoofHole.zip
    - Dominica_SintMaarten_TheBahamas_USVI_RoofDecking.zip
    - Dominica_SintMaarten_TheBahamas_USVI_RoofHole.zip

### Delineate roof decking and roof holes (dual-class model trained with data from Dominica and Sint Maarten)
1. Perform Steps 1-4 of *[Prepare Dominica training images](#Prepare-Dominica-training-images)* if you have not yet created a new ArcGIS Pro project, connected to the downloaded files folder, and set the folder as the default folder.
2. In the Catalog pane, expand the folder that contains the downloaded files, expand **Tools**, expand **Roof Damage Assessment.atbx**, and double-click **Delineate Roof Damage**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/4_DualClass_2.JPG?raw=true)

3. In the Delineate Roof Damage tool, for *Prepared Test Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Test_Images.gdb**, and select **OK**.
4. In the Delineate Roof Damage tool, for *Trained Model (Dual-Class)*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Dominica_SintMaarten_RoofDecking_RoofHole**, select **Dominica_SintMaarten_RoofDecking_RoofHole.emd (or .dlpk)**, and select **OK**.
5. In the Delineate Roof Damage tool, for *Output Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **New Item**, select **File Geodatabase**, type *Predicted_Polygons_Dominica_SintMaarten_RoofDecking_RoofHole*, press **Enter**, select the new file geodatabase, and select **OK**.
6. In the Prepare Images tool, for *Scratch File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files. If a scratch geodatabase has already been created in a previous step, select **Scratch.gdb** and select **OK**. If not, select **New Item**, select **File Geodatabase**, type *Scratch*, press **Enter**, select the new file geodatabase, and select **OK**.
7. In the Delineate Roof Damage tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/4_DualClass_7.JPG?raw=true)

8. While the tool is running, the progress bar indicates the current test image in which roof damage is being delineated. To track which images have been used and how many are left, select **View Details**.
9. Once the tool is done running, the predicted polygons file geodatabase is complete.

### Delineate roof decking (single-class model trained with data from Dominica and Sint Maarten)
1. With the Delineate Roof Damage tool still open, for *Trained Model (Dual-Class)*, delete the previously input file path.
2. In the Delineate Roof Damage tool, for *Trained Model (Roof Decking)*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Dominica_SintMaarten_RoofDecking**, select **Dominica_SintMaarten_RoofDecking.emd (or .dlpk)**, and select **OK**.
3. In the Delineate Roof Damage tool, for *Output Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **New Item**, select **File Geodatabase**, type *Predicted_Polygons_Dominica_SintMaarten_RoofDecking*, press **Enter**, select the new file geodatabase, and select **OK**.
4. In the Delineate Roof Damage tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/4_RoofDecking_4.JPG?raw=true)

5. Once the tool is done running, the predicted polygons file geodatabase is complete.

### Delineate roof decking (single-class model trained with data from Dominica, Sint Maarten, The Bahamas, and US Virgin Islands)
1. With the Delineate Roof Damage tool still open, for *Trained Model (Roof Decking)*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Dominica_SintMaarten_TheBahamas_USVI_RoofDecking**, select **Dominica_SintMaarten_TheBahamas_USVI_RoofDecking.emd (or .dlpk)**, and select **OK**.
2. In the Delineate Roof Damage tool, for *Output Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **New Item**, select **File Geodatabase**, type *Predicted_Polygons_Dominica_SintMaarten_TheBahamas_USVI_RoofDecking*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Delineate Roof Damage tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/4_ExpandedRoofDecking_3.JPG?raw=true)

4. Once the tool is done running, the predicted polygons file geodatabase is complete.

### Delineate roof holes (single-class model trained with data from Dominica and Sint Maarten)
1. With the Delineate Roof Damage tool still open, for *Trained Model (Roof Decking)*, delete the previously input file path.
2. In the Delineate Roof Damage tool, for *Trained Model (Roof Hole)*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Dominica_SintMaarten_RoofHole**, select **Dominica_SintMaarten_RoofHole.emd (or .dlpk)**, and select **OK**.
3. In the Delineate Roof Damage tool, for *Output Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **New Item**, select **File Geodatabase**, type *Predicted_Polygons_Dominica_SintMaarten_RoofHole*, press **Enter**, select the new file geodatabase, and select **OK**.
4. In the Delineate Roof Damage tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/4_RoofHole_4.JPG?raw=true)

5. Once the tool is done running, the predicted polygons file geodatabase is complete.

### Delineate roof holes (single-class model trained with data from Dominica, Sint Maarten, The Bahamas, and US Virgin Islands)
1. With the Delineate Roof Damage tool still open, for *Trained Model (Roof Decking)*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Dominica_SintMaarten_TheBahamas_USVI_RoofHole**, select **Dominica_SintMaarten_TheBahamas_USVI_RoofHole.emd (or .dlpk)**, and select **OK**.
2. In the Delineate Roof Damage tool, for *Output Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **New Item**, select **File Geodatabase**, type *Predicted_Polygons_Dominica_SintMaarten_TheBahamas_USVI_RoofHole*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Delineate Roof Damage tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/4_ExpandedRoofHole_3.JPG?raw=true)

4. Once the tool is done running, the predicted polygons file geodatabase is complete.

[Back to top](#Delineate-post-storm-roof-damage-using-deep-learning-and-aerial-imagery)

---

## 5. Evaluate deep learning models

### Download files
1. Go to the [file repository](https://drive.google.com/drive/folders/11AbOA5j1tVF8iIRS7vwDq8YJcy1Z3hhy).
2. Download and unzip the following files if they were not already downloaded or created in a previous step:
    - Tools.zip
    - Polygons.zip
    - Prepared_Test_Images.gdb.zip

### Evaluate dual-class model (trained with data from Dominica and Sint Maarten)
1. Perform Steps 1-4 of *[Prepare Dominica training images](#Prepare-Dominica-training-images)* if you have not yet created a new ArcGIS Pro project, connected to the downloaded files folder, and set the folder as the default folder.
2. In the Catalog pane, expand the folder that contains the downloaded files, expand **Tools**, expand **Roof Damage Assessment.atbx**, and double-click **Calculate Accuracy**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/5_DualClass_2.JPG?raw=true)

3. In the Calculate Accuracy tool, for *Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Predicted_Polygons_Dominica_SintMaarten_RoofDecking_RoofHole.gdb**, and select **OK**.
4. In the Calculate Accuracy tool, for *Reference Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Reference_Polygons.gdb**, and select **OK**.
5. In the Calculate Accuracy tool, for *Prepared Test Images File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **Prepared_Test_Images.gdb**, and select **OK**.
6. In the Calculate Accuracy tool, for *Output Accuracy Tables File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Accuracy_Tables_Dominica_SintMaarten_RoofDecking_RoofHole*, press **Enter**, select the new file geodatabase, and select **OK**.
7. In the Calculate Accuracy tool, for *Scratch File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files. If a scratch geodatabase has already been created in a previous step, select **Scratch.gdb** and select **OK**. If not, select **New Item**, select **File Geodatabase**, type *Scratch*, press **Enter**, select the new file geodatabase, and select **OK**.
8. In the Calculate Accuracy tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/5_DualClass_8.JPG?raw=true)

9. While the tool is running, the progress bar indicates which predicted polygons feature class is being evaluated. To track which feature classes have been evaluated and how many are left, select **View Details**.
10. Once the tool is done running, the accuracy tables file geodatabase is complete.

### Evaluate roof decking model (trained with data from Dominica and Sint Maarten)
1. With the Calculate Accuracy tool still open, for *Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Predicted_Polygons_Dominica_SintMaarten_RoofDecking.gdb**, and select **OK**.
2. In the Calculate Accuracy tool, for *Output Accuracy Tables File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Accuracy_Tables_Dominica_SintMaarten_RoofDecking*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Calculate Accuracy tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/5_RoofDecking_3.JPG?raw=true)

4. Once the tool is done running, the accuracy tables file geodatabase is complete.

### Evaluate roof decking model (trained with data from Dominica, Sint Maarten, The Bahamas, and US Virgin Islands)
1. With the Calculate Accuracy tool still open, for *Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Predicted_Polygons_Dominica_SintMaarten_TheBahamas_USVI_RoofDecking.gdb**, and select **OK**.
2. In the Calculate Accuracy tool, for *Output Accuracy Tables File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Accuracy_Tables_Dominica_SintMaarten_TheBahamas_USVI_RoofDecking*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Calculate Accuracy tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/5_ExpandedRoofDecking_3.JPG?raw=true)

4. Once the tool is done running, the accuracy tables file geodatabase is complete.

### Evaluate roof hole model (trained with data from Dominica and Sint Maarten)
1. With the Calculate Accuracy tool still open, for *Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Predicted_Polygons_Dominica_SintMaarten_RoofHole.gdb**, and select **OK**.
2. In the Calculate Accuracy tool, for *Output Accuracy Tables File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Accuracy_Tables_Dominica_SintMaarten_RoofHole*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Calculate Accuracy tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/5_RoofHole_3.JPG?raw=true)

4. Once the tool is done running, the accuracy tables file geodatabase is complete.

### Evaluate roof hole model (trained with data from Dominica, Sint Maarten, The Bahamas, and US Virgin Islands)
1. With the Calculate Accuracy tool still open, for *Predicted Polygons File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, double-click **Polygons**, select **Predicted_Polygons_Dominica_SintMaarten_TheBahamas_USVI_RoofHole.gdb**, and select **OK**.
2. In the Calculate Accuracy tool, for *Output Accuracy Tables File Geodatabase*, select the folder icon. In the new window, navigate to the folder that contains the downloaded files, select **New Item**, select **File Geodatabase**, type *Accuracy_Tables_Dominica_SintMaarten_TheBahamas_USVI_RoofHole*, press **Enter**, select the new file geodatabase, and select **OK**.
3. In the Calculate Accuracy tool, select **Run**.

![](https://github.com/maja-kucharczyk/roof-damage-assessment/blob/main/img/5_ExpandedRoofHole_3.JPG?raw=true)

4. Once the tool is done running, the accuracy tables file geodatabase is complete.

[Back to top](#Delineate-post-storm-roof-damage-using-deep-learning-and-aerial-imagery)
