# Telecom-Customer-Churn-Prediction
ML Classification problem to derive business insights and identify potential customers who might churn.
"Predict behavior to retain customers. Analyze all relevant customer data and develop focused customer retention programs." 
## Introduction
This is one of the Kaggle Datasets https://www.kaggle.com/blastchar/telco-customer-churn. Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
The data set includes information about:
*	Customers who left within the last month – the column is called Churn.
*	Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
*	Customer account information – how long they have been a customer, contract, payment method, paperless billing, monthly charges, and total charges.
*	Demographic info about customers – gender, age range, and if they have partners and dependents.
The goal of the project is to explore this type of models, learn more about the subject and come up with recommendations for customer retention program.
## Preprocessing
After loading data, I checked for duplicated and missing values. There were no duplicated values but the column ‘Total Charges’ had 11 missing rows. It turned out that those rows were for the customers with '0' month of tenure. That means that they are very new customers and have been using provided services for less than 1 month. So, it is too early for them to decide whether to stay with the provider or not and it was clearly seen that the values in column Churn for all of them is 'No'. As a result, the missing values were replaced with ‘0’.
## Exploratory Data Analysis
I started to analyze data with calculations of the churn rate.  The dataset contains 7043 rows which means 7043 customers. The number of customers that stayed with the company (no-churn) is 5174, and the number of customers that left (churn) is 1869. The churn rate is 26.5% which is quite a big number. More than a quarter of customers left the company within last month and taking care of this serious problem should be one of the main priorities in company’s strategy. Looking at the data closely should help with revealing the reasons of customer churn and thus developing retain program. 
Building a pairplots showing distribution of numeric variables for churned and non-churned customers revealed a tendency of the customers with higher monthly charges are more likely to leave the company comparing to the ones with lower monthly charges.

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/MonthlyCharges_tenure_churn.png)

Just building tenure distribution plot without dividing customers on non-churned and churned can already tell some facts about customers’ behavior and company itself:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/tenure_distrib.png)

We see two distinct spikes at the end of the charts. The spike on the left tells that there is a relatively big number of new customers 0-4 months meaning that company has a good ability to attract new customers. It looks like that marketing and sales departments of the company are working well. However, a big number of those new customers will leave the company during next months. In an ideal situation when there is no churn, tenure distribution chart should have a spike only at its right end, meaning that over the time number of customers is building up at the very end of the analyzed period. In our situation the spike at the right end tells that there are always customers who keep their loyalty since company was capable to satisfy them with the service all this time. 

Customer tenure Distribution Churn vs No Churn is showing it even clearer:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/tenure_distrib_churn.png)

Number of churned customers is significantly bigger than no churn for the first 20 months, but the peak of churn is at 0–4-month period and then number of churned customers decreasing over the time. That means that the company has no time for analyzing customers’ behavior and should act right away to retain clients because otherwise most of them will leave even after the first month. So, when a customer joins the company, we should be able to assess his/her risk of churn immediately. And we hope that the set of parameters (variables) that describe each customer will help us to identify patterns which lead customers to churn.

Let us go back to the parameter we already touched Monthly Charges and take a closer look at it. So far, we know that the customers with higher charges are more likely to churn. Here is the distribution of monthly charges:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/MonthlyCharges_distrib.png)

Looking at this chart we can visually combine customers into three groups. The low charges - 20-30$ per month, moderate charges – 40-60$ per month and high charges – 70-110$ per month. With regards to churn vs no churn monthly charges distribution is as follows:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/MonthlyCharges_distrib_churn.png)

The lowest number of churn customer is in the low monthly charges group, the highest number of churn customer is in the high monthly charges group. The moderate monthly charges group has roughly equal number of churned and no churned customers. So, a new customer that chose the plan with high monthly charges should fall into company’s highest attention priority. Because we can see that the high monthly charge group is the biggest group thus brining the company most of its earnings, but it is also the group with highest risk of customers to churn which means the company is losing good chunk of its revenue.

### Demographic
Demographic charts are showing no dependency on gender between churn and no churn groups of customers. However, we can see that customers who do not have partners and no dependents have as twice as big probability to churn comparing to those who have partners and dependents. Another interesting fact is showing the chart dividing customers by seniority parameter. Even though the percentage of senior customers is relatively small only 16.3% a little bit less than a half of them are churned customers. So, company should pay closer attention to single customers with no kids or any other dependents and to senior customers. There must be something about these groups of customers that puts them into higher risk to leave the company.

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Demographicpng_Page1.png)

### Services
Looking at the group of charts representing churned and no churned customers with regards to services we can see the following picture: 

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Servicespng_Page1.png)

There is no dependency on whether customer has or does not phone service and multiple lines. The number of churned customers in those groups is roughly ¼ which is the overall churn rate we figured out earlier.

The charts showing groups of customers with regards to types of internet service is revealing the fact that the customers who have fiber optic internet service have way higher churn rate ~40% comparing to DSL internet service – 18% and no internet service – 7%. So, the customers with fiber optic internet are going to be another subgroup that should be in company’s attention focus. The second chart showing distribution of monthly charges with regards to type of internet revealing that fiber optic is the service that associates with high monthly charges only, while DSL internet covers broad range of charges from the lowest to mid-high charges. ‘No internet service’ group associates only with the lowest charges.

### Additional Services
The group of charts dividing customers depending on additional services they chose to have or not to:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Additional%20Servicespng_Page1.png)

It is clearly seen that customers who chose not to have additional services have a drastically higher churn rate comparing to those ones who have them - 40% vs 15-20%. Here is another group of customers for the company to work with. The situation itself might look counterintuitive at first glance. Customers with additional services apparently have higher monthly charges but we already figured out that the customers with high monthly charges have higher churn rate. Explanation could be that the difference in monthly charges between customers with additional services and without them is not that significant but customers without additional services still fall into high monthly charges group. Lack of additional quite useful services like tech support, online security etc. cause inconvenience and issues for customers and make them think that service they have from the company not worth money they pay.

The distribution of monthly charges of churned customers without additional services chart is showing that customers from this group are in each subgroup in terms of monthly charges we defined earlier – low, moderate and high:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/MonthlyCharges_no_additional_services_distrib.png)

Most of this type of customers is in the high monthly charges group. Why customers who do not have additional services are still paying high monthly charges? Obviously there must be other parameters that effect monthly payments and not represented in our dataset. For example, speed of Internet. Choosing higher Internet speed and no additional services may not be that beneficial.
The only additional services that do not affect the churn rate are Streaming TV and Streaming movies. Regardless of them the churn rates both subgroups are roughly the same:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Streamingpng_Page1.png)

The other thing about all charts showing churn of customers with regards to additional services are customers with no internet service. Only 1.6% of all customers are churned customers who have no internet service. It cannot be compared to churn rates of any other subgroup of customers who have internet service. It leads to conclusion that the quality of internet service overall is not satisfying enough.

### Type of contract
The charts showing churned vs no churned customers depending on type of contract:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Contract_churn.png)

We see that most of churned customers 23.5% (26.6% overall churn rate) have a month-to month type of contract. Moreover, more than a half (55%) of all customers have this type of contract. This is another especially important factor for the company to work with. It looks like one-year and two-year types of contracts keep customers from leaving whereas a month-to-month contract provides a convenient opportunity for almost all not satisfied customers to not prolong their contract for the next month and leave. However, it is highly unlikely that month-to-month type of contract by itself, the way it is designed, causes churn of customers. More important is to figure out what are the reasons behind it that force customers to use this easy opportunity to leave.

### Billing and Payment method
Another subgroup of customers that requires closer attention is those who receive paperless billing:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/PaperlessBilling_churn.png)

The churn rate of this group is almost triple as big as customers receiving their bills in paper.

Let us look to the last parameter – payment method:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/PaymentMethod_churn.png)

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/MonthlyCharges_PaymentMethod_churn.png)

‘Electronic check’ group has 4 times bigger churn rate comparing to other groups. The second chart is showing that electronic check is not associated with high monthly charges by itself since we can see it covers all range from the lowest to the highest charges. Nevertheless, we do see a ’bubble’ at high monthly charges area. That likely means that customers who chose electronic check method of payment also tend to have services associated with higher charges like Fiber optic internet service. Anyway, what the issues are with ‘Electronic check’ method of payment is a subject of thorough investigation.

### Senior citizens
I went back to demographic groups. Earlier it was figured out that Senior customers is a group with churn rate ~40% which is 1.5 times higher than the overall churn rate. I selected senior customers only from the dataset and looked at all parameters. It was all the same trends as I saw earlier regardless demographic groups but now some subgroups of senior customers like ‘Month-to-month contract’, ‘Electronic check’ and some others have churn rate close or even over 50% which is twice higher than overall churn rate. It explains high churn rate in Senior customers demographic group in total.

#### Services

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Senior_InternetService.png)

#### Type of Contract

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Senior_Contract.png)

#### Billing and Payment method
![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Senior%20Billing%20and%20Payment%20methodpng_Page1.png)

## Modeling
Preparing data for Machine Learning (ML) I started with removing ‘Customer ID’ column as a unique customer identifier which will not help to predict customer churn, converting predictor variable ‘Churn’ into a binary numeric variable and getting dummies for all categorical variables. After that I also built a chart showing correlation of each variable with a target variable ‘Churn’:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Correlation_churn.png)

As we can see on this graph all positively correlated with ‘Churn’ variables are those ones we figured out during Exploratory Data Analysis. These are month-to-month contract, fiber optic internet, no additional services, electronic check, monthly charges, paperless billing, senior citizens, no partners and no dependents.

Due to numerical variables which are ‘tenure’, ‘Monthly charges’ and ‘Total charges’ have different scales and for better ML algorithms performance they should be either standardized or normalized. I chose standardization since it is less affected by outliers.

To assess a model performance there are different metrics can be used. We are dealing with imbalanced data which means one out of two possible outcomes is overweigh the other. In our case negative outcomes, in other words ‘Churn’=0, which means not churned customers are 73.5%. Choosing Accuracy as a metric in this case cannot be reliable since having a model that predicts all '0' will automatically get 73.5% accuracy but it is no use at all. There is also another aspect to take into consideration. A trade-off between False Positive (FP) and False Negative (FN) predicted results. In our case FP means number of customers identified by a model as going to churn customers but they are actually not, and FN means number of customers identified by a model as not going to churn however they actually are. We would like to optimize FN number since we are striving to target all potentially churn customers with our retain program, but FN is a number of customers that will not be covered by this program. In cases like this using the area under ROC curve as a metric is recommended.

Another technique that can improve a model performance in case of imbalanced data is applying different resample methods. I applied the next resample methods:
1.	Upsample. Randomly resampled the minority class creates new data.
2.	SMOTE. Creates new synthetic data by using nearest neighbors of the minority class.
3.	Downsample. Randomly removes observations of majority class.

In order to choose resampling technique, I trained Logistic Regression to each of them and applied 5-Fold cross validation to assess model performance. I got the next results:
*	The best ROC_AUC Score for Original training set: 0.840745950411631
*	The best ROC_AUC Score for Upsampled training set: 0.8423111716044671
*	The best ROC_AUC Score for SMOTE training set: 0.9281181977074635
*	The best ROC_AUC Score for Downsampled training set: 0.8477800472030514

SMOTE resample technique provided the best result during 5-Fold cross-validation. So, this technique I used to train models.

I trained and tested five ML algorithms that are known to work well for binary classification with the following results of chosen metric:
1.	Logistic Regression ROC_AUC score: 0.8602
2.	Random Forest Classifier ROC_AUC score: 0.8304
3.	Stochastic Gradient Descent (SGD) ROC_AUC score: 0.8601
4.	Gradient Boosting Classifier ROC_AUC score: 0.8574
5.	AdaBoost Classifier ROC_AUC score: 0.8581

As it seen they are close to each other, however Logistic Regression provided slightly better results than other algorithms.

The plot of the derived feature importances identified by Random Forest Classifier is just another confirmation of already known facts:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Feature_Importances.png)

Hyperparameters tuning for Logistic Regression using GridSearch method could only increase cross-validation ROC AUC score just a little bit but there was no improvement on the test set, so I left this algorithm with default parameters.

In attempt to improve model performance I tried to combine two different models that provided the best results so far – Logistic Regression and Stochastic Gradient Descent. As a result, I got a very slight improvement of ROC AUC score up to 0.8604 so this ensembled model was picked to be used for production.

To visualize model performance, I built ROC and Precision-Recall graphs:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/ROC%20Graph.png)

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Precision-Recall%20Graph.png)

Since ROC AUC score of the models differs at fourth digit after decimal point it makes them almost undistinguishable on ROC graph. At the same time models have the same Precision-Recall score which lump their curves together making basically one line one the graph.

Using pickle module, I saved model to a file to store it for future usage when new unlabeled data come.

# Retention Program. Recommendations. 
The goal of the project is not only to build a predictive model to identify customers who are likely to churn but also develop a retention program for them.

Although the dataset contains 20 variables describing each customer there is still some information that is not given but can be helpful to develop specific focused retain program. For example, we do not know how many plans exist and what are monthly charges for each of them, what technologies are used for fiber optic Internet service – PON, FTTH, FTTN, etc. Depending on these factors approaches to retain customers may vary. However, it is still possible to provide fairly detailed recommendations to some of company’s departments based on which focused retention programs can be developed.

Recommendations themselves are based on the insights revealed during Exploratory Data Analysis. The correlation plot is a good summarising illustration of what factors (variables) are associated with churn customers:

![](https://github.com/sivasrk002/Telecom-Churn-Prediction/blob/main/Pictures/Correlation_churn.png)

Variables that are above X axis are subjects to work with during developing retention programs.

## Technical factors. Fiber optic Internet service.
Exploratory analysis disclosed that there are many customers who found fiber optic internet service not satisfying. Since fiber optic provides the fastest internet connection among other technologies it is clearly showing that the company has technical issues with its fiber optic lines. There are three key factors that can be reasons for customers to be unhappy with this service:
1.	Speed
2.	Service interruptions 
3.	Down time
### Recommendations for the Technical Department:
1.	Check if the speed of internet connection for each customer corresponds with the one specified on their contract. If it does not, then conduct investigation to find out reasons. Possible steps to apply:
    1.	Check headend active equipment settings. Consider upgrading equipment if needed.
    2.	Check the signal light levels at a customer’s side. If they below the threshold, test the line to get OTDR (Optical time domain reflectometer) trace to find out insertion loss (IL) along the line. Consider re-splicing fibre at the splice points along the line, where IL is above 0.3 dBm.
    3.	Get a report about over-congested fibre enclosures along the line where micro bends cause high IL. Consider fixing micro bends of affected fibres or even upgrading enclosure itself if possible.
    4.	Clean all optical connectors at the headend and a customer side.
2.	Get a report about average number of nonintentional service interruptions (outages) per month.
    1.	Identify the cause of each outage whether it is at the headend or on the line in the field.
    2.	Most of outages happen in the field in over-congested fiber enclosures where fibers can get tangled thus create micro bend and break. If fibers in an enclosure cannot be reorganized or an enclosure itself cannot be upgraded, consider putting an enclosure under strict limited access (embargo) to fiber technicians during building new fiber connections or maintenance.
    3.	Make sure that all maintenance works that entail service interruption are conducted during ‘maintenance window’ only which is 2am-6am and with 2 days notice to all affected customers.
3.	Consider building a backup fiber line for all back-bone routes where possible in order to avoid service interruptions when it is an outage on a primary rout.
4.	When all previous steps are implemented properly it should solve most of the issues with speed, service interruptions and down time. However, still consider having 24/7 on call crew on rotation basis to minimize down time when an outage happens.

## Additional Services.
Additional services are Online Security, Online Backup, Device Protection, Tech Support. Customers who do not have these services are more likely to leave the company. The possible reason is that lack of additional services causes too much of inconvenience to the customers, so they decide to leave.
### Recommendations for the Sales/Marketing Departments:
1.	While signing a contract with a new customer emphasize benefits of additional services how helpful and useful they can be. Especially explain clearly benefits of Tech Support for senior customers who are not usually familiar with indoor telecommunication and network equipment like a modem and a Wi-Fi router. Sometimes guiding them how to reboot those devices can solve the problem.
2.	Consider revising price-formation for the plans with no additional services. Two opposite approaches can be implemented:
    1.	If possible consider making monthly charges lower so they are not in high charges group anymore and thus look more reasonable for customers.
    2.	Combine all additional services in one package and calculate prices for the plans with this package. Make monthly charges for the plans with no additional services lower but in the way that plans with additional services look beneficial and worth paying some extra money. Thus, gently forcing new customers to make choice towards plans with additional services.
3.	New customers who still choose plans with no additional services should be contacted by the company’s representative within first 3 months and receive the offer with the additional services package and clear explanations about its advantage. 
4.	Consider contacting current customers with no additional services and offer them to add those services to their plans providing explicit explanations about acquired benefits.

## Type of contract.
Exploratory Data Analysis showed a big churn rate among customers who have month-to-month type of contract. The possible reason is that this type of contract is the first opportunity for all unsatisfied with company’s services customers to leave the company by simply not prolonging the contract and it is most likely not related to its design or provisions.
### Recommendations for the Sales/Marketing Departments:
1.	Consider revising one-year and two-years types of contracts in the way looking them more attractive to the customers. For instance, lowering monthly charges for the same plans comparing to the month-to-month contract and as a result making one- and two-years contracts more beneficial.
2.	Consider developing six-months contract as an intermediate type with implementing of the same idea as in the previous step. 
3.	Those new customers who still choose month-to-month contract should be contacted by the company’s representative within first 3 months and receive the offer to switch to other types of contract with clear explanations of their benefits.

## Paperless Billing.
As it was revealed during Exploratory Analysis paperless billing is associated with high customer churn rate. The possible reason may relate to the way customers receive their invoices. For instance, emails with an invoice get to Spam folder, so customers are not aware about due date and as a result, those of them who do not have an automatic payment set up get penalized for a late payment.
### Recommendations for the IT Department:
1.	Make sure that billing emails optimized for all common email services like google, yahoo, hotmail etc. and do not get identified as spam. Remind customers to add company’s domain to trusted sources to avoid getting received emails to Spam folder.
2.	Consider having an online account for each customer which provides them possibility to download, collect and archive their statements if it is still not the case.

### Recommendations for the Sales Department:
1.	Dealing with new customers always ask about their preferences regarding type of billing. Especially do not propose paperless billing by default to senior people since most of them do not have much familiarity how to use emails and online accounts. For those senior customers who choose paperless billing make sure they know how it works and provide clear explanations if needed.
2.	Consider contacting current customers who have paperless billing and asking them if they are having any difficulties with their bills. For those of them who are confused and unhappy offer switching to paper bills. 

## Electronic check.
This type of payment is another factor related to churn customers.Even though electronic check provides some advantages to customers it is still associates with high fraudulent potential and processing errors.
### Recommendations to Accountant department:
1.	Ask for a report from the servicing bank about any fraud activities with customers electronic checks detected. If there were any how they were investigated and what means were applied to make transactions more secure.
2.	Ask for a report from the servicing bank about any processing errors like incorrect withdrawal, double withdrawal etc. and how fast they were detected and reimbursed to the customers.
3.	Based on these two reports make a conclusion if frauds and processing errors are main reasons for customers to be unhappy with electronic check type of payment.
### Recommendations for the Sales Department:
1.	For all types of plans and contracts consider specifying automatic bank transfer or automatic credit card charging as a default type of payment. It will provide more secure and fast transaction and prevent customers being penalized for late payments. 
2.	Still let new customers know about other possible types of payments but switch to them only if customers ask for it.
3.	Consider contacting current customers who pay their bills by electronic check and asking them to switch to any of other possible options.

## Conclusion.
Some of the recommendations above have a preventive nature and affect not only certain groups of customers but company’s operation itself. This is a result of the revealed fact that the company has technical issues with some of its important services like fiber optic lines, paperless billing and electronic check. Preventive measures by themselves should reduce the number of not satisfied customers and subsequently decrease the churn rate. Only after troubleshooting all technical issues and making sure that all services work properly recommendations for the Sales and Marketing department can be implemented.

In order to minimize spending company’s resources and maximize efficiency of retain programs all recommendations to Sales and Marketing Departments that entail contacting customers and making offers should be done with the priority to those customers who were identified by the model as potentially churn customers.

At the same time an approximate cost of implementing retain program to a single customer should be calculated. It would help management to assess trade-off between two opposite intentions:
*	To implement retain programs to a bigger number of customers trying to cover all False Negative customers – those who were identified by the model as no churn, but they will actually leave the company.
*	To save company’s resources by implementing retain programs only to those customers who are really going to leave the company, which is never known 100%.

Understanding this factor and knowing company’s limitations like number of employees who will be involved in conducting retain programs, the allocated budget for implementation of the retain programs, especially for resolving the technical issues as the most expensive but the most important ones, will help the company to reach its goal to reduce the churn rate to an acceptable level.
