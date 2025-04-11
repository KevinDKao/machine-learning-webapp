# Machine Learning Interactive Visualization Web App

An interactive web application built with Python and Dash that helps beginners understand fundamental machine learning algorithms through beautiful visualizations. The app features three key machine learning models with real-time parameter adjustments and visual feedback.

![Demo Screenshot](https://github.com/user-attachments/assets/06e6e7d0-1227-42d2-8e5b-3f1af3d7fc51)

## Features

- **K-Nearest Neighbors Classifier**: Visualize how the KNN algorithm classifies data points into categories and how the decision boundary changes with different k values.
- **K-Nearest Neighbors Regression**: See how KNN can be used for predicting continuous values and how the number of neighbors affects the prediction curve.
- **Logistic Regression**: Understand how logistic regression creates decision boundaries and how the regularization parameter impacts model complexity.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/machine-learning-webapp.git
cd machine-learning-webapp
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python dashapp.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:8050
```

3. Interact with the models:
   - Use the tabs to switch between different algorithms
   - Adjust parameters using the sliders
   - Observe how the models' behavior changes in real-time



## Acknowledgments

This was written and developed for Dr. Lorenzo Luzi of Rice University. Thank you for the opportunity to make fun tools for people to learn from! 
