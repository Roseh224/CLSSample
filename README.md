# CLSSamples

### Overview 
This repository contains snippets of project I completed during my time as an Assistant Technician on the SGM Beamline of the Canadian Light Source (CLS). During my time working at the CLS I was working to develop an application to add to the code module of my supervisor. It concerned process automation and data forecasting and the primary language for the project was Python.

### Description
The purpose of the application is rather specific, and it is as follows:

On the beamline I was working on, samples would be sent in to be analyzed. The purpose of the analysis was to determine the chemical makeup of a sample, and it was done by scanning the sample. Multiple scans of a sample would be needed, and the exact number of scans needed depended on the specific sample. This could become rather labor-intensive, because taking scans is time consuming, and since the number of scans needed varied, a beamline operator needed to be present in order to determine how many scans to take of the sample.

The application I developed utilized the information from an initial set of 5 scans of a sample. It examined these scans in order to predict how many additional scans would be needed to determine the chemical make-up of the sample. This greatly reduced the amount of labor required to analyze a sample, because an operator didn't need to be present for the entire length of the scanning process. The application is still in-use today, and the github account of the Canadian Light Source's SGM Beamline where I developed this application is as follows: https://github.com/Canadian-Light-Source/sgmdata/blob/master/sgmdata/utilities/predict_num_scans.py

### Tools and Technologies

The primary languages and technologies used for this project were Python, GitHub, Jupyter Notebooks, and EPICs. 
