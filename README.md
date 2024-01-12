# mobile_health_segmentation

## Introduction

Blood pressure measurements are vital in assessing overall health and are closely linked to cardiovascular health risks. With the global prevalence of cardiovascular diseases, accurate blood pressure monitoring is paramount. Traditional clinical measurements, while accurate, offer only a snapshot of an individual's blood pressure and may be influenced by the clinical environment. This limitation has led to the increasing importance of at-home blood pressure monitoring (HBPM) and ambulatory blood pressure monitoring (ABPM).

However, HBPM is often challenged by the user's ability to adhere to proper measurement techniques, which are crucial for obtaining reliable results. Incorrect practices in HBPM, such as improper cuff positioning or incorrect posture, can lead to inaccurate readings, diminishing the effectiveness of this monitoring method.

Our application leverages the functionalities of a laptop, including its camera, notifications, and microphone, to guide users through the blood pressure measuring process. This interactive approach aims to ensure adherence to the best practices for blood pressure measurement, including correct posture, arm position, cuff placement, and timing of measurements. By providing visual cues and feedback, the application seeks to educate users and promote consistent, accurate measurements.

This repository contains the code and resources for the Blood Pressure Monitoring Assistance Application. It is an innovative solution to a common public health challenge, aiming to make accurate blood pressure monitoring more accessible and effective for everyone.

This project aims to enhance the accuracy and reliability of at-home blood pressure measurements, a critical aspect of hypertension management. Our application is designed to assist individuals in complying with the best practices for blood pressure measurement, thereby improving the quality of data for both personal monitoring and clinical evaluation.


## Installation
To install requirements, please use venv and use: 

``` bash
pip install -r requirements.txt
```

Please install the approporiate `torch`, `torchvision` library for your operating system. 

To train the UNet, run the following command:
``` bash
python train.py
```

To run the application, use the following command:
``` bash
python main.py
```

Look into `test.ipynb` for sample outputs/testing.




