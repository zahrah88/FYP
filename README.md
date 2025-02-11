Student Performance Analysis – Final Year Project
📌 Project Overview
Education is one of the most critical aspects of personal and societal development, yet student performance varies widely based on a combination of personal, socioeconomic, and institutional factors. This project aims to analyze the key factors impacting student academic performance, with a specific focus on students for whom English is a second language (ESL).

By leveraging data analysis, visualization techniques, and machine learning, the project investigates how various factors—such as attendance, parental involvement, socioeconomic status, and access to resources—influence student outcomes. The insights generated will help educators, policymakers, and school administrators make data-driven decisions to improve academic performance, especially for ESL students.

The final deliverable of this project is an interactive dashboard that visualizes key findings and allows users to explore student performance trends based on different variables. Additionally, machine learning models will be developed to predict student success and identify at-risk students early on.

📊 Research Questions
This project seeks to answer the following key questions:

1️⃣ How do socioeconomic factors (e.g., family income, parental education, school type) affect student performance?
2️⃣ What is the impact of attendance and study habits (e.g., hours studied, extracurricular activities) on academic success?
3️⃣ How does English proficiency influence the performance of ESL students compared to native English speakers?
4️⃣ Can machine learning models accurately predict student performance based on key influencing factors?

📂 Project Structure
The project is organized into multiple directories to maintain clarity and reproducibility:

📂 datasets/        → Raw and cleaned datasets used in analysis  
📂 scripts/         → Python scripts for data processing, machine learning models, and dashboard creation  
📂 visuals/         → Graphs, charts, and visualizations
📂 dashboard/       → Power BI reports and Dash-based visualization tools  
📂 docs/            → Project reports, documentation, and presentations  
📂 datasets/        → This folder contains both raw datasets and cleaned/preprocessed versions. It includes:

Student_Performance.csv
StudentPerformanceFactors.csv
EDI_Dummy_Data.csv
Cleaned_Student_Performance.csv
Cleaned_StudentPerformanceFactors.csv
Cleaned_EDI_Dummy_Data.csv
notebooks/ – Jupyter notebooks used for:

Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Machine Learning Model Development
scripts/ – Python scripts for:

Data Cleaning & Preprocessing
Machine Learning Model Training & Evaluation
Data Visualization & Dashboard Creation
dashboard/ – Contains:

Power BI reports
Python Dash-based interactive dashboard
docs/ – Contains:

Project Proposal & Interim Progress Reports
Final Project Report
Presentation Slides
📊 Data Sources & Description
This project is based on three primary datasets, each contributing unique insights into student performance:

1️⃣ Student Performance Dataset
Contains academic scores, study habits, and attendance records
Features include:
Hours Studied
Previous Academic Scores
Attendance Percentage
Participation in Extracurricular Activities
Performance Index (aggregated measure of student success)
2️⃣ Student Performance Factors Dataset
Focuses on external influences on student success, such as:
Parental Education Level
Family Income
Access to Resources (e.g., internet, tutoring)
Peer Influence & School Type
Motivation Level
3️⃣ EDI (Equality, Diversity & Inclusion) Dummy Data
Provides demographic and equity-related attributes, including:
Ethnicity & Gender
Disability Status
Accommodation Type & Commuting Distance
Scholarship & Financial Aid Status
Each dataset underwent extensive preprocessing, including handling missing values, feature encoding, and standardization, to ensure consistency and accuracy in analysis.

🔍 Methodology & Workflow
The project follows a structured data science workflow:

1️⃣ Data Collection & Cleaning

Handled missing values via mean/mode imputation
Applied one-hot encoding & ordinal encoding for categorical variables
Used Min-Max Scaling & Standardization for numerical variables
2️⃣ Exploratory Data Analysis (EDA)

Conducted univariate & multivariate analysis
Generated correlation heatmaps, scatter plots, histograms, and boxplots
Used ANOVA tests & regression analysis to identify significant predictors
3️⃣ Feature Engineering & Machine Learning Model Development

Selected key predictors using Recursive Feature Elimination (RFE)
Trained and compared models:
✅ Linear Regression (baseline model)







