# Banking---Vehicle-Loan

### Project Overview:
Banks and financial institutions are incurring significant losses due to vehicle loan defaults — when borrowers fail to repay their loans. This has led to the tightening up of vehicle loan underwriting and an increase in vehicle loan rejection rates. The need for a better credit risk scoring model is also raised by these institutions. This warrants a study to estimate the determinants of vehicle loan default.

### Insights:
- A Zipped Dataset & Data Dictionary file can be found [here](Dataset)
- Python script can be found [here](Vehicle%20Loans.ipynb)

### Approach:
- Understand why people default: identify key factors and uncover patterns and insights that differentiate defaulters from non-defaulters.
- Train a machine learning model to assist banks in making better loan approval decisions.

### Tasks & Observations - Vehicle Loan Default Prediction

##### Data Understanding, Cleaning, & Preprocessing
- Handled missing values using appropriate imputations.
- Corrected inconsistent data types (converted dates, numeric fields).
- Normalized category values (e.g., score descriptions).
- Removed unnecessary or redundant fields.
- Scanned or binned skewed numeric variables for better model performance.

##### Exploratory Analysis

1. **Employment Type & Default Behavior**
   
The analysis shows that self-employed borrowers default slightly more than salaried borrowers, but the difference is minor (22.7% vs. 20.3%).
This suggests employment type is not a strong standalone predictor.

2. **Target Variable Imbalance**

The data is heavily skewed toward non-defaulters (78% non-default vs. 22% default).
This imbalance directly caused early logistic regression models to classify almost everything as “non-default.”

3. **Branch, Supplier & Manufacturer Analysis**

- Default Rate by Top 10 Branch_id
Branch_id 36 shows the highest default rate at around 29%, indicating the loans associated with this branch have the highest risk of default, followed by branch_id 16 at around 28% of default risk. On the lower side, branch_id 19 indicates the lowest risk of default at around 16%.

- Default Rate by Top 10 State_id

State_id 13 has the highest default rate at around 31%, indicating this state’s borrowers are more likely to default compared to others. The second highest default rate is for State_id 14, with nearly 28%, also showing elevated risk. Other states such as State_id 8, 9, 6, 7, 4, 5, 3, and 1 show moderate default rates ranging between roughly 17.5% to 23%. State_id 1 has the lowest default rate among these top 10, around 17.5%, indicating relatively better loan performance in this region.

- Default Rate by Supplier_id
  
Supplier_id 18317 shows the highest default rate, around 33.7%, indicating that loans associated with this supplier have the highest risk of default. Supplier_id 21980 and 15694 also exhibit relatively high default rates, around 30% and 29.7%, respectively, suggesting these suppliers' borrowers tend to default more frequently. 

On the lower end, Supplier_id 14375 has the lowest default rate at about 11.7%, indicating better credit performance for loans from this supplier. Other suppliers like 14234, 15663, and 14145 have default rates in the 22-25% range, reflecting moderate risk.

Supplier_id 18166 stands out with a notably low default rate of approximately 15.1%, suggesting better borrower quality or more effective risk management practices.
  
- Default Rate by Manufacturer_id

Manufacturer_id 153 exhibits the highest default rate, approximately 33.5%, indicating this manufacturer’s associated loans carry the greatest risk of default. Several manufacturers, including 48, 45, 49, 51, 67, 86, and 120, show default rates clustered in the 20-27% range, reflecting moderate default risk.

Notably, Manufacturer_id 152 shows a default rate of zero or close to zero, which is an anomaly and warrants further investigation—this could be due to insufficient data, misclassification, or truly excellent repayment performance.

The manufacturer with the lowest default rate (excluding 152) is 145, at around 20%, suggesting better loan performance relative to others.

4. **Age and Default Patterns**

Ages 31-45 seem to be most active, which makes sense - they are typically in their prime working years, with stable income. Younger (<19) and older (>61) borrowers may have fewer loans and potentially lower default rates. This could be due to lower borrowing activity, more conservative borrowing behavior, or financial support structures (like parents, or pensions).

This implies that while age has an impact, it might not be linear. Consider interaction terms with other features (income, employment type, etc.) could improve predictive model.

9. **ID Proof Usage**

As Mobile No. has the record of 233154, customers in dataset provided, this proof ID seems to be mandatory for the loan approval. The Adhar card is the most popular choice among customers, followed by Voter Id and PAN card. Passport is the least common proof of ID.

10. **Credit Score (CNS) as a Risk Indicator**


11. **Primary vs. Secondary Account Behavior**


12. **Sanctioned vs. Disbursed Amount**


13. **Number of Inquiries**


14. **Credit History Features**

  
15.  **Predictive Modeling Observations**


### Executive Summary – Vehicle Loan Default Analysis

This project analyzes over 233,000 vehicle loan applications to identify the key factors influencing loan default and support the development of a predictive risk model. The dataset is highly imbalanced, with 78% non-defaulters and 22% defaulters, requiring careful handling for modeling.

The analysis shows that credit-related variables are the strongest predictors of default, while demographics and identity information contribute only marginally. Borrowers with high-risk or very-high-risk credit scores show default rates above 27–30%, whereas low-risk groups default at only 15–18%. Customers with no credit history default at a relatively high rate of 23%, indicating that thin-file customers have elevated risk.

Credit exposure plays a significant role: borrowers with zero primary accounts but multiple secondary accounts show the highest default rate (55%), highlighting the danger associated with co-signed or secondary loans without direct repayment responsibility. Defaults gradually decrease as customers accumulate more primary account history.

Demographic factors such as age and employment type show limited predictive power. Self-employed borrowers default slightly more than salaried borrowers, but the difference is not operationally meaningful. The most active borrower age group (31–45) shows moderate risk.

Sanctioned vs. disbursed amounts show extreme skewness, and inquiry count has almost no relationship with default, contrary to typical credit industry expectations.

Overall, the findings reinforce that credit scoring, account history, and borrower credit depth are the most reliable indicators of default risk. Traditional logistic regression performed poorly due to class imbalance and linear constraints, indicating the need for non-linear ML models such as Random Forest or XGBoost. These insights provide a strong foundation for building a robust vehicle loan risk prediction system that can help lenders reduce NPAs and refine underwriting strategies.


