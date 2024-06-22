# Synthetic Data Science Project

***Project Overview***

This project demonstrates the process of generating a synthetic dataset, performing basic descriptive and inferential statistics, and visualizing the data using Python. The dataset includes information on Age, Height, Weight, Gender, and Income for 1000 samples. The analysis is conducted using the following libraries: Pandas, NumPy, Matplotlib, and Seaborn.

***Libraries Used***

- Pandas: For data manipulation and analysis.
- NumPy: For numerical operations and generating synthetic data.
- Matplotlib: For creating static visualizations.
- Seaborn: For creating advanced statistical visualizations.

***Data Generation***

The synthetic dataset is generated with the following specifications:

- Age: Normally distributed with a mean of 35 and a standard deviation of 10.
- Height: Normally distributed with a mean of 170 cm and a standard deviation of 15 cm.
- Weight: Normally distributed with a mean of 70 kg and a standard deviation of 10 kg.
- Gender: Randomly assigned with 50% probability for 'Male' and 'Female'.
- Income: Normally distributed with a mean of 50,000 and a standard deviation of 15,000.

***Descriptive Statistics***


The following descriptive statistics were calculated for Age, Height, Weight, and Income:

- Mean
- Median
- Standard Deviation
- Variance
- The mode was calculated for the Gender column.

***Data Visualization***


The distributions and potential outliers in the dataset were visualized using:

- Histograms: For Age, Height, Weight, and Income.
- KDE Plots: For Age, Height, Weight, and Income using Seaborn.
- Boxplots: For Age, Height, Weight, and Income to identify outliers.

***Correlation Analysis***


The Pearson correlation coefficients were calculated between Age, Height, Weight, and Income to understand the relationships between these variables.

***Hypothesis Testing***


A t-test was performed to determine if there was a significant difference in Income between Males and Females. The results were:

- t-statistic: 0.161
- p-value: 0.872
- The high p-value indicated that there was no statistically significant difference in Income between Males and Females.

***Insights and Conclusions***

Descriptive Statistics: Provided a summary of central tendencies and variability for Age, Height, Weight, and Income.
Mode for Gender: Indicated that 'Male' was the most frequent gender in the dataset.
Visualizations: Helped in understanding the distributions and identifying potential outliers.
Correlation Analysis: Showed the relationships between the variables.
Hypothesis Testing: Concluded that there was no significant difference in Income between Males and Females, as indicated by the high p-value.

***How to Run the Project***

Ensure you have Python installed on your system.
Install the required libraries:

pip install pandas numpy matplotlib seaborn
Run the Python script containing the project code.

***Code Snippets***


***Library Setup***

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

***Data Generation***

```
np.random.seed(0)
num_samples = 1000
age = np.random.normal(loc=35, scale=10, size=num_samples)
height = np.random.normal(loc=170, scale=15, size=num_samples)
weight = np.random.normal(loc=70, scale=10, size=num_samples)
gender = np.random.choice(['Male', 'Female'], size=num_samples)
income = np.random.normal(loc=50000, scale=15000, size=num_samples)
```

***Create DataFrame***

```
df = pd.DataFrame({
    'Age': age,
    'Height': height,
    'Weight': weight,
    'Gender': gender,
    'Income': income
})
print(df.head())
```

***Descriptive Statistics***

```
descriptive_stats = df.describe()
print(descriptive_stats)

gender_mode = df['Gender'].mode()[0]
print('Mode for Gender:', gender_mode)
```

***Data Visualization***

```
sns.histplot(df['Age'], kde=True)
plt.show()
sns.histplot(df['Height'], kde=True)
plt.show()
sns.histplot(df['Weight'], kde=True)
plt.show()
sns.histplot(df['Income'], kde=True)
plt.show()
```

```
sns.boxplot(data=df[['Age', 'Height', 'Weight', 'Income']])
plt.show()
Correlation Analysis
python
Copy code
correlation_matrix = df[['Age', 'Height', 'Weight', 'Income']].corr()
print(correlation_matrix)
```

***Hypothesis Testing***

```
from scipy.stats import ttest_ind

income_male = df[df['Gender'] == 'Male']['Income']
income_female = df[df['Gender'] == 'Female']['Income']
t_test_result = ttest_ind(income_male, income_female)
print(t_test_result)
```
