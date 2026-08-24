# pH'ocus - Colorimetric pH Prediction Tool

An automated machine learning application for precise pH prediction from colorimetric indicators. This tool bridges analytical chemistry and digital image processing to provide rapid, objective, and reproducible pH measurements directly from a smartphone or desktop.



https://github.com/user-attachments/assets/9877b209-b6b8-420b-ab6f-bb0ade26ad98



## The Problem
<img width="364" height="383" alt="Capture d&#39;écran 2026-08-24 191756154" src="https://github.com/user-attachments/assets/bd1c0d7d-2bd5-4205-899a-0a836bc45fdd" />

While colorimetric pH indicators (such as test strips or liquid reagents) offer rapid, low-cost screening, their interpretation remains fundamentally limited by human subjectivity. 

In rigorous chemical engineering and analytical workflows, relying on visual comparison against a reference scale introduces critical challenges:
* **Subjectivity and Inconsistency:** Different operators perceive color transitions differently, leading to inter-user variability.
* **Environmental Variables:** Ambient lighting and background contrast severely impact the perceived RGB profile of the indicator.
* **Lack of Traceability:** Manual readings are difficult to digitize, preventing seamless integration into laboratory information management systems (LIMS) or automated batch records.
* **Compromised Accuracy:** For tight pH specifications, the resolution of human vision is insufficient to interpolate values between standard color swatches reliably.

## The Solution

pH'ocus replaces the subjective visual estimation bottleneck with a quantitative, algorithmic workflow. By ingesting standard digital photographs, the application automates color extraction and maps RGB profiles to exact pH values using trained regression models. 

The application automatically corrects image orientation and isolates the region of interest using channel variance. It then applies an ensemble of machine learning models to the mean RGB array, providing a robust consensus prediction of the sample's pH. This ensures analytical reproducibility and allows for instant, objective characterization directly in the field or the lab.

## Core Capabilities

* **Ensemble Machine Learning:** Leverages multiple trained regressors (K-Nearest Neighbors, Support Vector, Random Forest, and Decision Tree) simultaneously to mitigate the bias of single-model predictions.
* **Smart ROI Detection:** Implements a localized variance algorithm to automatically detect the analytical target (e.g., the reactive pad), with real-time dynamic adjustment via UI or keyboard controls.
* **Real-time Colorimetric Translation:** Instantly translates the mean RGB values of the selected area into quantitative pH measurements.
* **Automated Image Optimization:** Automatically scales high-resolution images, corrects EXIF orientation, and processes data strictly in RAM to ensure low latency and minimal server load.
* **Cross-Platform Accessibility:** Built on a responsive framework, allowing immediate deployment and usage on any device equipped with a camera.
