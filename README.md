# Banking-Customer-Churn-Prediction-and-Retention-Strategy
## Project Overview and Purpose of the Project

This project delivers a comprehensive solution for predicting and strategizing against customer churn within a banking environment. It integrates machine learning, and business intelligence to identify at-risk customers, understand their churn drivers, and provide actionable insights for targeted retention efforts. The workflow encompasses data acquisition (conceptually from PostgreSQL), rigorous data preprocessing and model development in Python, and compelling interactive visualization in Power BI. The dataset is sourced from Kaggle. The primary purpose of this project is to equip a bank with the tools and insights necessary to combat customer churn effectively. This involves:
* **Early Churn Detection:** Building a predictive model to identify customers most likely to churn *before* they actually leave.
* **Driver Identification:** Uncovering the key factors and customer characteristics that contribute to churn.
* **Strategic Intervention:** Providing data-driven recommendations and actionable customer segments for targeted retention campaigns.
* **Performance Monitoring:** Creating an interactive dashboard to monitor churn trends.

## Technologies Used

* **Python 3.x**
* **PostgreSQL** 
* **Power BI**

## Methodology
### Step 1: Data Acquisition & Initial Exploration

* **Purpose:** To load the raw customer data into the Python environment and perform initial checks to understand its structure, quality, and basic statistics.
* **Procedure:**
    * **Conceptual PostgreSQL Role:** In a real banking environment, this stage would involve connecting to a PostgreSQL database (or similar data warehouse) using SQL queries (e.g., `SELECT * FROM customer_data WHERE active_status = 'Y'`) to extract the raw customer dataset. The `Churn_Modelling.csv` file used here represents this extracted data.
    * **Loading Data:** The `pandas` library is imported, and the `Churn_Modelling.csv` file is loaded into a DataFrame (`df = pd.read_csv('Churn_Modelling.csv')`). The output confirms: `Dataset 'Churn_Modelling.csv' loaded successfully.`
    * **Initial Inspection:** `df.head()` is used to display the first 5 rows, providing a quick visual glance at the data and column names.
    * **Data Information:** `df.info()` provides a concise summary, showing data types and non-null counts. A critical finding from its output is that there are **10,000 entries with no missing values** across all columns, simplifying subsequent preprocessing.
    * **Descriptive Statistics:** `df.describe()` generates summary statistics for numerical columns. Key observations from its output include:
        * The mean of the `Exited` column is `0.2037`, indicating an **overall churn rate of approximately 20.37%** in the dataset.
        * The 25th percentile for `Balance` is `0.00`, suggesting **at least a quarter of the customers have zero balance**, which could be a significant factor in churn.
* **Insights:** The dataset is clean and complete, requiring no missing value imputation. The initial statistics provide a clear picture of the churn problem's scale and highlight initial areas of interest (e.g., zero balance accounts).

### Step 2: Data Preprocessing & Feature Engineering

* **Purpose:** To clean and transform the raw data into a numerical format suitable for machine learning algorithms, removing irrelevant information and encoding categorical variables.
* **Procedure:**
    * **One-Hot Encoding:** Categorical features such as `Geography` and `Gender` are converted into numerical format using one-hot encoding (`df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True, dtype=int)`). The `drop_first=True` argument prevents multicollinearity by ensuring that one category can be inferred from the absence of others (e.g., if not 'Female', then 'Male'). The `dtype=int` ensures the new columns are integers (0s and 1s).
* **Insights:** The dataset is now prepared for machine learning, free from non-predictive identifiers and with all categorical data appropriately converted to numerical form.

### Step 3: Machine Learning Model Development & Training

* **Purpose:** To set up the machine learning pipeline by splitting the data into training and testing sets, scaling numerical features, and then training the chosen classification model.
* **Procedure:**
    * **Feature and Target Separation:** The independent features (`X`) are separated from the target variable (`y`, which is the `Exited` column). This clearly defines the inputs and outputs for the model.
    * **Feature Scaling:** Numerical features in both training and test sets are scaled using `StandardScaler`. `scaler.fit_transform(X_train)` fits the scaler *only* on the training data and transforms it, while `scaler.transform(X_test)` uses the *same* fitted scaler to transform the test data. This prevents data leakage from the test set and ensures features with larger magnitudes (e.g., `EstimatedSalary`) don't disproportionately influence the model.
    * **Model Initialization & Training:** A `LogisticRegression` model is imported and initialized (`model = LogisticRegression(random_state=42, solver='liblinear')`). The `solver='liblinear'` is chosen for its suitability with smaller datasets and L1/L2 regularization. The model is then trained (`model.fit(X_train_scaled, y_train)`) using the scaled training features and their corresponding churn labels.
* **Insights:** A robust machine learning framework is established. The data splitting and scaling procedures ensure that the model is trained and evaluated in a way that accurately reflects its performance on unseen customer data.

### Step 4: Model Evaluation & Validation

* **Purpose:** To rigorously assess the performance of the trained Logistic Regression model in predicting churn, utilizing standard classification metrics.
* **Procedure:**
    * **Predictions:** The model generates both binary class predictions (`y_pred = model.predict(X_test_scaled)`) and, more critically, **churn probabilities (`y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]`)** for the test set. Probabilities are essential for nuanced risk assessment and segmenting customers based on their likelihood of churning.
    * **Metrics Import:** Necessary evaluation functions (`accuracy_score`, `classification_report`, `confusion_matrix`) are imported from `sklearn.metrics`.
    * **Performance Report:**
        * The `accuracy_score` is calculated and printed: `Accuracy: 0.80`. While 80% accuracy seems high, it can be misleading in imbalanced datasets.
    * **Confusion Matrix (Cell 17):** The `confusion_matrix` is printed, showing the counts: `[[1520 73], [324 83]]`. This directly reveals:
        * **True Negatives (1520):** Customers correctly identified as non-churners.
        * **False Positives (73):** Customers incorrectly identified as churners (loyal customers wrongly flagged).
        * **False Negatives (324):** **Critical Insight:** Customers who *actually churned* but were predicted by the model *not* to churn (missed opportunities for intervention).
        * **True Positives (83):** Customers correctly identified as churners.
* **Insights:** The model performs reasonably well overall, but its key limitation is the low `recall` for the churn class. This means a significant number of actual churners (324 in the test set) are being missed by the current model. This insight is crucial for understanding the model's practical utility and directing future improvements.

### Step 5: Generating Churn Predictions & Customer Risk Segments

* **Purpose:** To apply the trained model to the *entire* customer base, calculate a churn probability for each customer, and then categorize these customers into actionable risk segments.
* **Procedure:**
    * **All Customer Probabilities:** The `StandardScaler` (which was fitted on the training data in Step 3) is applied to the *entire* `X` dataset to ensure consistent scaling. The trained Logistic Regression model then calculates the `Churn Probability` for every single customer (`all_customer_churn_probabilities = model.predict_proba(all_customer_scaled)[:, 1]`).
    * **Risk Segmentation:** The calculated `Churn Probability` is added as a new column to the original DataFrame (`df['Churn Probability'] = all_customer_churn_probabilities`). Custom thresholds (e.g., `high_risk_threshold = 0.7`, `medium_risk_threshold = 0.3`) are then defined and used to categorize each customer into a `Churn Risk Segment` (High Risk, Medium Risk, Low Risk). The `df['Churn Risk Segment'].value_counts()` output shows the distribution of customers within these segments (e.g., `Low Risk: 9546, Medium Risk: 445, High Risk: 9`).
* **Insights:** This step transforms raw model outputs into direct business intelligence. By assigning customers to clear risk segments, the bank gains an immediate, actionable understanding of its customer base's churn likelihood, allowing for tailored intervention strategies and efficient resource allocation.

### Step 6: Interactive Business Intelligence with Power BI

* **Purpose:** To create an intuitive and interactive dashboard that visualizes churn patterns, key drivers, and actionable customer lists, enabling business stakeholders to easily understand and act upon the analytical findings.
* **Explanation:** The final DataFrame, now enriched with `Churn Probability` and `Churn Risk Segment` columns, is exported to a CSV file named `customer_churn_insights_for_powerbi.csv`. This CSV file serves as the primary data source for the Power BI dashboard, ensuring that all derived insights are readily available for visualization. The notebook confirms successful export.
* **Insights:** This final step makes the complex analytical results accessible. The Power BI dashboard provides a dynamic interface for business users to filter, drill down, and gain deeper insights, fostering data-driven decision-making for churn prevention.

## Detailed Insights from Power BI Dashboard
#### **Dashboard Page 1: Executive Churn Overview**

* **Insights:**
The immediate highlight of the `20.4%` churn rate provides a clear top-level understanding of the attrition problem facing the bank.
Despite being a small percentage (`0.09%` or 9 customers) of the total customer base, the "High Risk" segment represents a critical and immediately actionable group.
The high "Avg Balance" for these high-risk customers quantifies the significant financial value at stake, underscoring the urgency and potential ROI of targeted retention efforts for this specific group.

#### **Dashboard Page 2: Churn Driver Analysis**
* **Insights:**
The visuals clearly indicate that customers with **"1 Product" exhibit the highest absolute churn**, suggesting they have fewer ties to the bank and are easier to lose. Conversely, customers with **"2 Products" often show the lowest churn rates**, highlighting a sweet spot for customer stickiness. A notable insight can be a *spike* in churn for customers with **"3+ Products"**, which might indicate dissatisfaction due to complexity or perceived value for money.
**"Inactive Members" consistently have a significantly higher churn rate** than "Active Members," emphasizing the importance of ongoing engagement.
* **Age Group Dynamics** Insights into which "Age Group" contributes the most to churn (e.g., 40-49 might have the highest *volume* of churn, while perhaps older groups might have higher *rates*).
* **Geographic Variations:** Identifies specific "Geography" regions that have disproportionately higher churn rates, suggesting a need for localized retention strategies or product adjustments.
* 
#### **Dashboard Page 3: High-Risk Customer Identification**
* **Insights:**
Provides a precise list of individual customers who are most likely to churn. This transforms abstract probabilities into concrete leads for intervention.
By showing individual attributes like `Balance` and `NumOfProducts`, relationship managers can understand the specific context of each high-risk customer before making contact, enabling personalized approaches.
The breakdown of high-risk customers by `Geography` (e.g., "6 out of 9 high-risk customers are from France") pinpoints specific areas where immediate, localized intervention is required, indicating potential regional issues.

#### **Dashboard Page 4: Predictive Insights & Trend Analysis**
* **Insights:**
The `Key Influencers` visual directly quantifies the impact of various factors on churn probability. For example, it can reveal "Number of products is more than 3" makes a customer **5.03x more likely to churn**, or "CreditScore <= 404" makes them **4.95x more likely to churn**. This provides highly granular, data-backed evidence for decision-making.
The `Decomposition Tree` allows for interactive, hierarchical drill-down into churned segments (e.g., exploring "Churned Customers" -> by "Inactive Status" -> by "Gender" -> by "Geography"). This helps identify complex multi-factor reasons for churn within specific micro-segments.
Trend analysis by `Tenure` (how long a customer has been with the bank) can highlight specific periods where customers are most vulnerable to churn (e.g., early churn within the first year, or churn spikes after 5+ years), guiding lifecycle-specific retention programs.

## Retention Strategy Recommendations

Based on the comprehensive insights derived from this project, here are actionable recommendations:

1.  **Targeted High-Value Interventions:** Prioritize immediate, personalized outreach (e.g., direct calls from relationship managers, tailored premium offers) to customers identified in the `High Risk` segment, especially those with high account balances.
2.  **Strategic Cross-Selling Campaigns:** Design and launch focused campaigns to encourage customers with only one product to acquire a second, perhaps through attractive bundles or exclusive benefits, thereby increasing their "stickiness." For customers with 3+ products, consider offering simplified financial reviews or bundled service benefits to address potential complexity issues.
3.  **Proactive Re-engagement Programs:** Implement systematic programs to re-engage `Inactive Members`. This could include personalized communications, exclusive offers for reactivating accounts, or feedback surveys to understand their reasons for disengagement.
4.  **Financial Health & Loyalty Initiatives:** Develop specific loyalty programs or financial advisory services for customers with lower credit scores or those with zero balances, aimed at improving their financial well-being and strengthening their relationship with the bank.
5.  **Geographically & Demographically Tailored Strategies:** Allocate additional resources and develop specific marketing and retention strategies for regions (e.g., France) and age groups identified as high-churn hotspots.
6.  **Continuous Monitoring & Model Refinement:** Regularly use the Power BI dashboard to monitor churn trends and evaluate the effectiveness of implemented retention campaigns. Use this real-world feedback to iteratively improve the underlying machine learning model (e.g., by exploring new features, models, or imbalance handling techniques).


