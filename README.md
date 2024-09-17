# Predicting NBA Game Outcomes

## Description
This project addresses a binary classification problem, aiming to predict NBA game outcomes by applying 
suitable machine learning models. The analysis exclusively uses the 2021 NBA schedule dataset, 
without incorporating player-specific statistics. To improve prediction accuracy, 
various data mining techniques are employed, including data preprocessing, feature engineering, feature selection, 
and hyperparameter tuning. Several classifiers were trained and tested using cross-validation, 
including Random Forest, Decision Trees, Logistic Regression, and K-Nearest Neighbors. 
Logistic Regression performed best, achieving an accuracy of 63%, 
and was further used to predict outcomes for the 2022 season, obtaining a 61% accuracy rate. 

The rationale for this analysis is that by focusing on non player-specific data allows this research to highlight how 
variables other than individual skill have an impact on games. It is also interesting how team’s winning or losing their 
previous games have an impact on their winning percentage.

## Tech Stack
-  **Programming Language**: Python
-  **Libraries**:
    - Pandas (for data manipulation and analysis)
    - Matplotlib, seaborn (for data visualization and plotting)
    - Sciki-learn (for implementing and testing machine learning models)
- **Algorithms**:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - SVM

## Features
- Data cleaning and preprocessing of NBA game data.
- Feature engineering to enhance prediction accuracy.
- Implementation of four different machine learning algorithms.
- Comparison of algorithm performance using accuracy metrics and visual plots.
- Easy-to-read visualizations for better model evaluation and insights into the prediction process.
