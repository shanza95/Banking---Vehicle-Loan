# Banking---Vehicle-Loan

## Project Overview:
Banks and financial institutions are incurring significant losses due to vehicle loan defaults — when borrowers fail to repay their loans. This has led to the tightening up of vehicle loan underwriting and an increase in vehicle loan rejection rates. The need for a better credit risk scoring model is also raised by these institutions. This warrants a study to estimate the determinants of vehicle loan default.

## Insights:
- A Zipped Dataset & Data Dictionary file can be found [here](Dataset)
- Python script can be found [here](Vehicle%20Loans.ipynb)

## Approach:
- Understand why people default: identify key factors and uncover patterns and insights that differentiate defaulters from non-defaulters.
- Train a machine learning model to assist banks in making better loan approval decisions.

## Tasks & Observations - Vehicle Loan Default Prediction

### Data Understanding, Cleaning, & Preprocessing
- Handled missing values using appropriate imputations.
- Corrected inconsistent data types (converted dates, numeric fields).
- Normalized category values (e.g., score descriptions).
- Removed unnecessary or redundant fields.
- Scanned or binned skewed numeric variables for better model performance.

### Exploratory Analysis

1. **Employment Type & Default Behavior**
   
      The analysis shows that self-employed borrowers default slightly more than salaried borrowers, but the            difference is minor (22.7% vs. 20.3%).
      This suggests employment type is not a strong standalone predictor.

2. **Target Variable Imbalance**

      The data is heavily skewed toward non-defaulters (78% non-default vs. 22% default).
      This imbalance directly caused early logistic regression models to classify almost everything as “non-            default.”

3. **Branch, Supplier & Manufacturer Analysis**

   - *Default Rate by Top 10 Branch_id*
  
     Branch_id 36 shows the highest default rate at around 29%, indicating the loans associated with this branch       have the highest risk of default, followed by branch_id 16 at around 28% of default risk. On the lower side,      branch_id 19 indicates the lowest risk of default at around 16%.

   - *Default Rate by Top 10 State_id*

     State_id 13 has the highest default rate at around 31%, indicating this state’s borrowers are more likely to      default compared to others. The second highest default rate is for State_id 14, with nearly 28%, also             showing elevated risk. Other states such as State_id 8, 9, 6, 7, 4, 5, 3, and 1 show moderate default rates       ranging between roughly 17.5% to 23%. State_id 1 has the lowest default rate among these top 10, around           17.5%, indicating relatively better loan performance in this region.

   - *Default Rate by Supplier_id*
  
      Supplier_id 18317 shows the highest default rate, around 33.7%, indicating that loans associated with this        supplier have the highest risk of default. Supplier_id 21980 and 15694 also exhibit relatively high default       rates, around 30% and 29.7%, respectively, suggesting these suppliers' borrowers tend to default more             frequently. 

      On the lower end, Supplier_id 14375 has the lowest default rate at about 11.7%, indicating better credit          performance for loans from this supplier. Other suppliers like 14234, 15663, and 14145 have default rates         in the 22-25% range, reflecting moderate risk.

      Supplier_id 18166 stands out with a notably low default rate of approximately 15.1%, suggesting better            borrower quality or more effective risk management practices.
  
   - *Default Rate by Manufacturer_id*

      Manufacturer_id 153 exhibits the highest default rate, approximately 33.5%, indicating this manufacturer’s        associated loans carry the greatest risk of default. Several manufacturers, including 48, 45, 49, 51, 67,         86, and 120, show default rates clustered in the 20-27% range, reflecting moderate default risk.

      Notably, Manufacturer_id 152 shows a default rate of zero or close to zero, which is an anomaly and               warrants further investigation—this could be due to insufficient data, misclassification, or truly                excellent repayment performance.

      The manufacturer with the lowest default rate (excluding 152) is 145, at around 20%, suggesting better loan       performance relative to others.

4. **Age and Default Patterns**

   Ages 31-45 seem to be most active, which makes sense - they are typically in their prime working years, with      stable income. Younger (<19) and older (>61) borrowers may have fewer loans and potentially lower default         rates. This could be due to lower borrowing activity, more conservative borrowing behavior, or financial          support structures (like parents, or pensions).

   This implies that while age has an impact, it might not be linear. Consider interaction terms with other          features (income, employment type, etc.) could improve predictive model.

5. **ID Proof Usage**

   As Mobile No. has the record of 233154, customers in dataset provided, this proof ID seems to be mandatory for    the loan approval. The Adhar card is the most popular choice among customers, followed by Voter Id and PAN        card. Passport is the least common proof of ID.

6. **Credit Score (CNS) as a Risk Indicator**

   Very Low Risk shows the lowest and most expected default rate, followed by Low Risk, which also performs          safely. The Inactive group has an unexpectedly low default rate, suggesting an unusual but relatively safe        segment. Medium Risk and the “insufficient history” not-scored group show similar, moderate risk levels,          while High Risk and especially Very High Risk exhibit the highest default rates as expected. 

   Customers with no bureau history fall in the mid-risk range, slightly riskier than low-risk groups. This group    needs a close attention to find the cause of the outcome, which might be due to new-to-credit customers, or       unscored due to data mismatch.

   Other not-scored categories generally fall into the moderate range, with the “no update in 36 months” group       performing better than expected. The “50+ active accounts” category shows a perfect non-default rate, but this    is almost certainly due to a very small and unreliable sample.

7. **Primary vs. Secondary Account Behavior**

   Customers with 0 primary accounts show a very high default rate (56%), indicating a risky profile—likely          borrowers with no independent credit history who still appear on multiple secondary accounts, suggesting          financial exposure without full repayment responsibility. Across the rest of the table, default rates             generally range from 11% to 28%, decreasing as the number of primary accounts increases, and often remaining      low or decreasing with fewer secondary accounts. The bottom-right cell is zero, which likely means there were     no observed defaults or no data in that highest primary-and-secondary-account combination.

8. **Sanctioned vs. Disbursed Amount**

   | Account Type           | Mean Sanctioned Amount | Mean Disbursed Amount |
   | ---------------------- | ---------------------- | --------------------- |
   | **Primary Accounts**   | 218,503.9              | 218,065.9             |
   | **Secondary Accounts** | 7,295.9                | 7,180.0               |

   Based on the mean values, there is only a very small difference between the  amounts sanctioned and the          amounts actually disbursed for both primary and secondary account loans, indicating minimal deduction or          adjustment between approval and actual disbursement. However, the distributions are highly skewed, especially     for primary loans. This skewness is clear from the median being zero and the large standard deviation,            indicating that while most customers receive little or no sanctioned amount, a small number receive very large    amounts.

9. **Number of Inquiries**

   The correlation between the number of inquiries and loan default is 0.0437, indicating a very weak positive       relationship. This means the number of inquiries on its own has almost no linear predictive power for default.    However, the effect of inquiries may still appear in non-linear patterns or in combination with other factors     such as income, credit history, or existing loan exposure, which is common in credit risk modelling.

10. **Credit History Features**

   | Feature                         | NEW_ACCTS_IN_LAST_6M | DELINQUENT_ACCTS_IN_6M | CREDIT_HISTORY_LENGTH |       AVERAGE_ACCT_AGE | loan_default |
   | ------------------------------- | -------------------- | ---------------------- | --------------------- | ---   ------------- | ------------ |
   | **NEW_ACCTS_IN_LAST_6M**        | 1.0000               | 0.1828                 | 0.2001                |       0.0334           | -0.0294      |
   | **DELINQUENT_ACCTS_IN_LAST_6M** | 0.1828               | 1.0000                 | 0.2622                |       0.1713           | 0.0345       |
   | **CREDIT_HISTORY_LENGTH**       | 0.2001               | 0.2622                 | 1.0000                |       0.8320           | -0.0421      |
   | **AVERAGE_ACCT_AGE**            | 0.0334               | 0.1713                 | 0.8320                |       1.0000           | -0.0248      |
   | **loan_default**                | -0.0294              | 0.0345                 | -0.0421               |       -0.0248          | 1.0000       |

- *New accounts and delinquent accounts in the last 6 months* are moderately correlated (0.18), suggesting customers opening more new accounts may also have more delinquencies.
- *Credit history length* and *average account age* are highly correlated (0.83), as expected—they both reflect the length of credit experience.
- All four features have very weak correlations with loan default (all correlations close to zero, between -0.04 and 0.03), indicating none alone strongly predict default.
- Slight positive correlation between *delinquent accounts* and *default* (0.0345) aligns with intuition but remains very weak.
- Slight negative correlation of *credit history length* and *average account age* with *default* suggests longer histories marginally reduce default risk.

11.  **Predictive Modeling Observations**

<img width="886" height="804" alt="image" src="https://github.com/user-attachments/assets/f25a9712-f664-4ca2-943e-d221076db199" />

All feature correlations with loan_default are very close to zero (around 0.03 or -0.03), indicating no strong linear relationship between these categorical or ID-like features and loan default.

Features like branch_id, supplier_id, Current_pincode_ID, and Employment_Type show some moderate correlations among themselves (up to about 0.21), which suggests possible grouping or clustering patterns but not directly predictive of default.

UniqueID, manufacturer_id, and loan_default have near-zero correlations with other features, meaning they are largely independent.

Since all feature correlations with the target (loan_default) are very weak, these features alone are unlikely to be strong predictors in a linear model.

This suggests that either more relevant features need to be included, or that non-linear models and feature engineering (e.g., encoding categories better, creating interaction terms) will be necessary to improve predictive power.

Features like IDs and categorical variables may need to be transformed carefully (e.g., target encoding, embeddings) rather than used as-is.

### Executive Summary – Vehicle Loan Default Analysis

This project analyzes over 233,000 vehicle loan applications to identify the key factors influencing loan default and support the development of a predictive risk model. The dataset is highly imbalanced, with 78% non-defaulters and 22% defaulters, requiring careful handling for modeling.

The analysis shows that credit-related variables are the strongest predictors of default, while demographics and identity information contribute only marginally. Borrowers with high-risk or very-high-risk credit scores show default rates above 27–30%, whereas low-risk groups default at only 15–18%. Customers with no credit history default at a relatively high rate of 23%, indicating that thin-file customers have elevated risk.

Credit exposure plays a significant role: borrowers with zero primary accounts but multiple secondary accounts show the highest default rate (55%), highlighting the danger associated with co-signed or secondary loans without direct repayment responsibility. Defaults gradually decrease as customers accumulate more primary account history.

Demographic factors such as age and employment type show limited predictive power. Self-employed borrowers default slightly more than salaried borrowers, but the difference is not operationally meaningful. The most active borrower age group (31–45) shows moderate risk.

Sanctioned vs. disbursed amounts show extreme skewness, and inquiry count has almost no relationship with default, contrary to typical credit industry expectations.

Overall, the findings reinforce that credit scoring, account history, and borrower credit depth are the most reliable indicators of default risk. Traditional logistic regression performed poorly due to class imbalance and linear constraints, indicating the need for non-linear ML models such as Random Forest or XGBoost. These insights provide a strong foundation for building a robust vehicle loan risk prediction system that can help lenders reduce NPAs and refine underwriting strategies.


