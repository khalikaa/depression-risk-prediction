# Project Overview
This dashboard is a final project for a Data Mining course, developed to identify the most influential factors that contribute to the risk of depressive disorder. Using data from the BRFSS survey, the project groups features into four categories—demographics, medical conditions, lifestyle, and social determinants—and applies a HistGradientBoosting machine learning model to analyze which factors have the strongest impact on depression risk.

## Web Interface
### Home Page
![Home](https://github.com/user-attachments/assets/2d3351ac-79e1-489d-a664-7d5b3e0e3da4)

### Discovery Page
![Discovery](https://github.com/user-attachments/assets/2c722619-5a62-4d0e-879f-003cec10968b)


### Prediction Page
![Prediction](https://github.com/user-attachments/assets/77696e7f-4ee3-4562-82dd-7a7ffba690fd)
#### Positive Prediction
![Prediction Result Positive](https://github.com/user-attachments/assets/26e2d6a2-e084-46e7-9e60-541de2682f19)
#### Negative Prediction
![Prediction Result Negative](https://github.com/user-attachments/assets/5944d82f-df32-44e4-80e2-eebf705dad1a)

<hr>

#### This dashboard is built with Flask and can be run locally using the following steps:

1. (Optional) Activate virtual environment:<br>
   `python -m venv venv`<br>
   `source venv/bin/activate` (for macOS/Linux)<br>
   `venv\Scripts\activate` (for Windows)

2. Install dependencies:<br>
   `pip install -r requirements.txt`

3. Run the application:<br>
   `python app.py`

The app will run at `http://localhost:5000`.
