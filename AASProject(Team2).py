#!/usr/bin/env python
# coding: utf-8

# In[220]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[221]:





# In[222]:


#Uploading files in dataframes df1 and df2


# In[224]:


df1 = pd.read_csv(r'C:\Users\SADIYA NAAZ ANSARI\Desktop\previous_application.csv')


# In[225]:


df2 = pd.read_csv(r'C:\Users\SADIYA NAAZ ANSARI\Desktop\application_data .csv')


# In[226]:


#checking 1st top 5 records of df1
df1.head()


# In[227]:


#checking 1st top 5 records of df2
df2.head()


# In[228]:


#checking no.of rows and columns of df1
df1.shape


# In[229]:


#checking no.of rows and columns of df2
df2.shape


# In[230]:


#checking columns
df1.columns


# In[231]:


#checking datatypes of df1
df1.dtypes


# In[232]:


#checking datatypes of df2
df2.dtypes


# In[233]:


df2.columns


# In[234]:


#Checking the numeric variables of the dataframes
df2.describe()


# In[235]:


#checked all the columns details
df2.info("all")


# In[18]:


# we understood that 122 columns ans 307511 columns


# In[236]:


#checking null values of df2
df2.isnull().sum()


# In[237]:


df2.head()


# In[21]:


#DATA CLEANING


# In[ ]:


# NULL VALUES


# In[238]:


#checking how many null values are present in each of the columns

#creating a function to find null values for the dataframe
def null_values(df2):
    return round((df2.isnull().sum()*100/len(df2)).sort_values(ascending = False),2)


# In[239]:


null_values(df2)


# In[ ]:


#Dealing with Null values more than 50%


# In[240]:


#creating a variable null_col_50 for storing null columns having missing values more than 50%

null_col_50 = null_values(df2)[null_values(df2)>50]


# In[27]:


#revieving null_col_50

print(null_col_50)
print()
print("Num of columns having missing values more than 50% :",len(null_col_50))


# In[ ]:


#CONCLUSION :
There are 41 columns having null values more than 50% which are related to different area sizes on 
apartment owned/rented by the loan applicant


# In[241]:


# Checking columns names having null values more than 50%
null_col_50.index  


# In[242]:


# Now lets drop all the columns having missing values more than 50% that is 41 columns

df2.drop(columns = null_col_50.index, inplace = True)


# In[243]:


df2.shape  


# In[ ]:


# Now there are 81 columns remaining......means 41 columns are dropped....still we are not sure about the desired columns 
which we want in our analysis....


# In[ ]:


#So now we will consider those columns having null values more than 15% for effiecient result


# In[244]:


# now we will deal with null values more than 15% 

null_col_15 = null_values(df2)[null_values(df2)>15]


# In[245]:


null_col_15


# In[ ]:


from the columns dictionary we can conclude that only 'OCCUPATION_TYPE', 'EXT_SOURCE_3 looks relevant to TARGET column.
thus dropping all other columns except 'OCCUPATION_TYPE','EXT_SOURCE_3


# In[246]:


#removing 'OCCUPATION_TYPE', 'EXT_SOURCE_3' from "null_col_15" so that we can drop all other at once.

null_col_15.drop(["OCCUPATION_TYPE","EXT_SOURCE_3"], inplace = True)


# In[247]:


print(null_col_15)
print()
print("No of columns having missing values more than 15% and are not reletable:",len(null_col_15))


# In[248]:


#thus removing columns having missing values more than 15% and which are not reletable to TARGET column
df2.drop(null_col_15.index,axis=1, inplace = True)


# In[249]:


df2.shape


# In[ ]:


#After after dropping 8 columns we are left with 73 columns
#There are 2 more Columns with missing values more than 15% ie. 'OCCUPATION_TYPE','EXT_SOURCE_3


# In[250]:


null_values(df2).head(10)


# In[ ]:


# Analyse & Removing Unneccsary Columns
# starting with EXT_SOURCE_3 , EXT_SOURCE_2. As they have normalised values, now we will understand the relation between 
these columns with TARGET column using a heatmap


# In[251]:


plt.figure(figsize= [10,7])

sns.heatmap(df2[["EXT_SOURCE_3","EXT_SOURCE_2","TARGET"]].corr(), cmap="Reds",annot=True)

plt.title("Correlation between EXT_SOURCE_3, EXT_SOURCE_2, TARGET", fontdict={"fontsize":20}, pad=25)
plt.show()


# In[ ]:


#There seems to be no linear correlation and also from columns description we decided to remove these columns.
#Also we are aware correation doesn't cause causation


# In[252]:


#dropping above columns as decide
df2.drop(["EXT_SOURCE_3","EXT_SOURCE_2"], axis=1, inplace= True)


# In[253]:


df2.shape


# In[254]:


#Using null_value function, checking null_values in %
null_values(df2).head(10)


# In[ ]:


# Now we will check columns with FLAGS and their relation with TARGET columns to remove irrelevant ones


# In[255]:


flaglist=[x for x in df2.columns if "FLAG" in x] #list created with columns which are having column names starts with FLAG word
flaglist


# In[256]:


# creating flag_df dataframe having all FLAG columns and TARGET column

flag_df = df2[flaglist+["TARGET"]]
flag_df


# In[257]:


import warnings
warnings.filterwarnings('ignore')


# In[258]:


# replacing "0" as repayer and "1" as defaulter for TARGET column

flag_df["TARGET"] = flag_df["TARGET"].replace({1:"Defaulter", 0:"Repayer"})


# In[56]:


flag_df["TARGET"]


# In[259]:


flag_df


# In[260]:


# as stated in columnn description replacing "1" as Y being TRUE and "0" as N being False

for i in flag_df:
    if i!= "TARGET":
        flag_df[i] = flag_df[i].replace({1:"Y", 0:"N"})


# In[261]:


flag_df


# In[264]:


import itertools


# In[265]:


# Plotting all the graph to find the relation and evaluting for dropping such columns

plt.figure(figsize = [20,24])

for i,j in itertools.zip_longest(flaglist,range(len(flaglist))):
    plt.subplot(7,4,j+1)
    ax = sns.countplot(flag_df[i], hue = flag_df["TARGET"], palette = ["r","b"])
    #plt.yticks(fontsize=8)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(i)


# In[ ]:


CONCLUSION :

Columns (FLAG_OWN_REALTY, FLAG_MOBIL ,FLAG_EMP_PHONE, FLAG_CONT_MOBILE, FLAG_DOCUMENT_3) have more repayers than defaulter 
and from these keeping FLAG_DOCUMENT_3,FLAG_OWN_REALTY, FLAG_MOBIL more sense thus we can include these columns and remove 
all other FLAG columns for furhter analysis.


# In[269]:


# removing required columns from "flag_df" such that we can remove the irrelevent columns from "df2" dataset.

flag_df.drop(["TARGET","FLAG_OWN_REALTY","FLAG_MOBIL","FLAG_DOCUMENT_3"], axis=1 , inplace = True)


# In[270]:


len(flag_df.columns)


# In[218]:


# dropping the columns of "flag_df" dataframe that is removing more 25 columns from "appl_data" dataframe

df2.drop(flag_df.columns, axis=1, inplace= True)


# In[272]:


df2.shape


# In[71]:


# Now we are left 46 revelent columns


# In[ ]:


#IMPUTING VALUES
Now that we have removed all the unneccesarry columns, we will proced with imputing values for relevent missing 
columns whereever required


# In[72]:


null_values(df2).head(10)


# In[ ]:


CONCLUSION :

Now we have only 7 columns which have missing values more than 1%. Thus, we will only impute them for further analysis and 
such columns are: OCCUPATION_TYPE, AMT_REQ_CREDIT_BUREAU_YEAR, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_HOUR


# In[74]:


#Replace null values with "Unknown"

df2["OCCUPATION_TYPE"] = df2["OCCUPATION_TYPE"].fillna("Unknown") 


# In[75]:


df2["OCCUPATION_TYPE"].isnull().sum() # Now we have zero null values 


# In[80]:


# Plotting a percentage graph having each category of "OCCUPATION_TYPE"

plt.figure(figsize = [12,7])
(df2["OCCUPATION_TYPE"].value_counts()).plot.bar(color= "orange",width = .8)
plt.title("Percentage of Type of Occupations", fontdict={"fontsize":20}, pad =30)
plt.show()


# In[ ]:


#Now let's move to other 6 columns :
AMT_REQ_CREDIT_BUREAU_YEAR", "AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_HOUR"


# In[208]:


df2[["AMT_REQ_CREDIT_BUREAU_YEAR","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_WEEK",
"AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_HOUR"]].describe()


# In[ ]:


#These above columns represent number of enquries made for the customer(which should be discrete and not continous). 
# from above describe results we see that all values are numerical and can conclude that for imputing missing we should not 
# use mean as it is in decimal form, hence for imputing purpose we will use median for all these columns.


# In[209]:


#creating "amt_credit" variable having these columns "AMT_REQ_CREDIT_BUREAU_YEAR","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_WEEK",
#"AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_HOUR"

amt_credit = ["AMT_REQ_CREDIT_BUREAU_YEAR","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_WEEK",
"AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_HOUR"]


# In[211]:


#filling missing values with median values

df2.fillna(df2[amt_credit].median(),inplace = True)


# In[212]:


null_values(df2).head(10)


# In[ ]:


# Still there some missing value columns but we will not impute them as the missing value count very less.


# In[ ]:


# Standardising values


# In[213]:


df2.describe()


# In[ ]:


# Conclusion:

from above describe result we can see that

columns AMT_INCOME_TOTAL, AMT_CREDIT, AMT_GOODS_PRICE have very high values, thus will make these numerical columns in categorical columns for better understanding.
columns DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, DAYS_LAST_PHONE_CHANGE which counts days have negative values. thus will correct those values
convert DAYS_BIRTH to AGE in years , DAYS_EMPLOYED to YEARS EMPLOYED


# In[215]:


# I can check the number of unique values is a column
# If the number of unique values <=40: Categorical column
# If the number of unique values in a columns> 50: Continuous

df2.nunique().sort_values()


# In[216]:


df2.shape


# In[217]:


df2.columns


# In[88]:


#changing negative ages to positive ages.
df2['DAYS_BIRTH']=abs(df2['DAYS_BIRTH'])
df2['DAYS_BIRTH'].describe()


# In[89]:


#changing negative values in days to positive days
df2['DAYS_EMPLOYED']=abs(df2['DAYS_EMPLOYED'])
df2['DAYS_EMPLOYED'].describe()


# In[90]:


#changing negative days to positive days.
df2['DAYS_REGISTRATION']=abs(df2['DAYS_REGISTRATION'])
df2['DAYS_REGISTRATION'].describe()


# In[91]:


#changing negative days to positive 
df2['DAYS_ID_PUBLISH']=abs(df2['DAYS_ID_PUBLISH'])
df2['DAYS_ID_PUBLISH'].describe()


# In[92]:


#converting the data type of categorical column
df2['REG_REGION_NOT_LIVE_REGION'] = df2['REG_REGION_NOT_LIVE_REGION'].astype(object)
df2.dtypes


# In[93]:


#Changing region from int to object
df2['REG_REGION_NOT_WORK_REGION'] = df2['REG_REGION_NOT_WORK_REGION'].astype(object)


# In[94]:


#Changing region from int to object
df2['LIVE_REGION_NOT_WORK_REGION'] = df2['LIVE_REGION_NOT_WORK_REGION'].astype(object)


# In[95]:


#Changing city from int to object
df2['REG_CITY_NOT_LIVE_CITY'] = df2['REG_CITY_NOT_LIVE_CITY'].astype(object)


# In[96]:


#Changing city from int to object
df2['REG_CITY_NOT_WORK_CITY'] = df2['REG_CITY_NOT_WORK_CITY'].astype(object)


# In[97]:


#Changing city from int to object
df2['LIVE_CITY_NOT_WORK_CITY']=df2['LIVE_CITY_NOT_WORK_CITY'].astype(object)


# In[98]:


df2.head()


# In[ ]:


# Handling Outliers
Major approaches to the treat outliers:

Imputation
Deletion of outliers
Binning of values
Cap the outlier


# In[99]:


#AMT_ANNUITY variable
#describe the AMT_ANNUITY variable of df

df2.AMT_ANNUITY.describe()


# In[100]:


sns.boxplot(df2.AMT_ANNUITY)
plt.title('Distribution of Amount Annuity')
plt.show()


# In[ ]:


As we take a look at AMT_ANNUITY column we can see that there are outliers at 258025. But there is no much differece between 
the mean and median, We can impute the outliers with Median here


# In[ ]:


#AMT_INCOME variable


# In[101]:


df2.AMT_INCOME_TOTAL.describe()


# In[102]:


plt.figure(figsize=(9,2))
sns.boxplot(df2.AMT_INCOME_TOTAL)
plt.xscale('log')
plt.title('Distribution of Income')
plt.show()


# In[ ]:


#AMT_CREDIT variable


# In[104]:


df2.AMT_CREDIT.describe()


# In[105]:


plt.figure(figsize=(9,2))
sns.boxplot(df2.AMT_CREDIT)
plt.title('Distribution of Credit amount')
plt.show()


# In[ ]:


#DAYS_BIRTH variable


# In[106]:


df2.DAYS_BIRTH.describe()


# In[107]:


sns.boxplot(df2.DAYS_BIRTH)
plt.title('Distribution of Age in the form of days')
plt.show()


# In[ ]:


#Conclusion:
DAYS_BIRTH column we can see from box plot that there are no outliers. There is no much difference between mean and 
median. Which means that all the applications received from the customers are of almost same age.


# In[ ]:


#DAYS_EMPLOYED variable


# In[111]:


df2.DAYS_EMPLOYED.describe()


# In[112]:


plt.figure(figsize=(15,5))
sns.boxplot(df2.DAYS_EMPLOYED)
#plt.yscale('log')
plt.title('Distribution of Days the client employed')

plt.show()


# In[ ]:


#Binning Continuous Variable


# In[ ]:


#AMT_INCOME_TOTAL variable


# In[113]:


#Creating bins for Credit amount

bins = [0,350000,700000,1000000000]
slots = ['Low','Medium','High']

df2['AMT_CREDIT_RANGE']=pd.cut(df2['AMT_CREDIT'],bins=bins,labels=slots)


# In[115]:


# Creating bins for income amount

bins = [0,200000,400000,10000000000]
slot = ['Low','Medium','High']

df2['AMT_INCOME_RANGE']=pd.cut(df2['AMT_INCOME_TOTAL'],bins,labels=slot)


# In[116]:


# Creating bins for days_birth

bins = [0,7300,10950,14600,18250,21900,25500]
slot = ['0-20','20-30','30-40','40-50','50-60','60-70']

df2['AGE_RANGE']=pd.cut(df2['DAYS_BIRTH'],bins,labels=slot)


# In[117]:


#Checking bin columns created in df.
df2.head()


# In[ ]:


#Analysis
#Checking the imbalance Percentage


# In[118]:


# Calculating Imbalance percentage
100*(df2.TARGET.value_counts())/ (len(df2))


# In[ ]:


#So TARGET column has 8.07% of 1's which means 8% clients have payment difficulties and 91.92% are having no difficulties


# In[119]:


# Dividing the dataset into two dataset of  target=1(client with payment difficulties) and target=0(all other)
target_1 = df2[df2['TARGET']==1]
target_0 = df2[df2['TARGET']==0]


# In[120]:


#Dataframe having target values 0
target_0.head()


# In[121]:


#Dataframe having target values 1
target_1.head()


# In[ ]:


# Univariate Analysis for target =0 and target=1


# In[ ]:


#Numeric variable
#Age


# In[122]:


# Numeric variable analysis for target_0 & target_1 dataframe
plt.figure(figsize = (15, 8))
plt.subplot(2, 2, 1)
plt.ylim(0,100000)
plt.title('Target=0 : Age-No Payment issues')
sns.countplot(target_0['AGE_RANGE'])

# subplot 2
plt.subplot(2, 2, 2)
plt.title('Target =1 : Age-Payment issues')
plt.ylim(0,100000)
sns.countplot(target_1['AGE_RANGE'])
plt.show()


# In[ ]:


# We can observe that customers belonging to age group 30-40 are able to make payment on time and can be considered 
while lending loan!
#The customers from 40 to 60 age are also can be considered.
# We can observe that customers belonging to age group 30-40 are more who ha spayment issues 


# In[ ]:


#Amount credit range


# In[123]:


# Numeric variable analysis for target_0 & target_1 dataframe
plt.figure(figsize = (15, 8))
plt.subplot(2, 2, 1)
plt.ylim(0,100000)
plt.title('Credit amount of loan - No payment issues')
sns.countplot(target_0['AMT_CREDIT_RANGE'],palette='muted')

# subplot 2
plt.subplot(2, 2, 2)
plt.title('Credit amount of loan- Payment issues')
plt.ylim(0,100000)
sns.countplot(target_1['AMT_CREDIT_RANGE'], palette='muted')
plt.show()


# In[ ]:


#Customers with less credit and most likely to make payment. Customers having medium and high credit can also be 
considered while lending the loan


# In[124]:


# Categorical Variable
# Occupation_type


# In[125]:


# Categorical variable analysis for target_0 & target_1 dataframe
plt.figure(figsize = (15,6))
plt.subplot(1, 2, 1)
#plt.subplots_adjust(wspace=0.5)


sns.countplot(target_0['OCCUPATION_TYPE'])
plt.title('Target=0 : Job type- no payment issues')
plt.ylim(0,50000)
plt.xticks(rotation = 90)

# subplot 2
plt.subplot(1, 2, 2)

sns.countplot(target_1['OCCUPATION_TYPE'])
plt.title('Target=1 : Job type- Payment issues')
plt.ylim(0,50000)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


# The plot clearly shows that labourers are most likely to make payment on time whereas HR staff are less likely to 
 make payment on time


# In[ ]:


# Name _Income _Type


# In[133]:


# Categorical variable analysis for target_0 & target_1 dataframe
plt.figure(figsize = (15,6))

plt.subplot(1, 2, 1)

sns.countplot(target_0['NAME_INCOME_TYPE'].dropna())
plt.title('Target=0 : Income type of people with no payment issues')
plt.ylim(0,150000)
plt.xticks(rotation = 90)

# subplot 2
plt.subplot(1, 2, 2)

sns.countplot(target_1['NAME_INCOME_TYPE'].dropna())
plt.title('Target=1 : Income type of  people with Payment issues')
plt.ylim(0,150000)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


#The plot clearly shows that working income type people are most likely to make payment on time whereas State Servant are less 
likely to make payment on time


# In[ ]:


# Analyse continuous column with respect to the target column


# In[ ]:


# Credit Amount


# In[148]:


#Analyse continuous column with respect to the target column
sns.distplot(target_0['AMT_CREDIT'], hist = False, label="Good")# Target = 0
sns.distplot(target_1['AMT_CREDIT'], hist = False, label='Bad')# Taget = 1
plt.title('AMT_CREDIT')
plt.legend()
plt.grid(color='black', linestyle='-', linewidth=0.25) 
plt.show()


# In[ ]:


#Amount Annuity


# In[152]:


#Analyse continuous column with respect to the target column
sns.distplot(target_0['AMT_ANNUITY'], hist = False, label="Good")# Target = 0
sns.distplot(target_1['AMT_ANNUITY'], hist = False, label="Bad")# Taget = 1
plt.title('AMT_ANNUITY')
plt.legend()
plt.grid(color='black', linestyle='-', linewidth=0.25) 
plt.show()


# In[153]:


# Goods price


# In[155]:


#Analyse continuous column with respect to the target column
sns.distplot(target_0['AMT_GOODS_PRICE'], hist = False,label= "good")# Target = 0
sns.distplot(target_1['AMT_GOODS_PRICE'], hist = False, label="bad")# Taget = 1
plt.title('AMT_GOODS_PRICE')
plt.legend()
plt.grid(color='black', linestyle='-', linewidth=0.25) 
plt.show()


# In[ ]:


# Analyse Categorical variables with respect to Target variable


# In[156]:


#Plot mutiple categorical columns with respect to Target column: Subplot
features = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']
list(enumerate(features))


# In[164]:


features = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']
plt.figure(figsize = (20, 40))

plt.subplots_adjust(hspace=0.8)
for i in enumerate(features):
    plt.subplot(5, 2, i[0]+1)
    sns.countplot(x = i[1], hue = 'TARGET', data = df2)
    plt.xticks(rotation = 45)


# In[ ]:


# From the above plot we can see that,

# Female customers pay loan amount on time and banks can target more female cusytomers for lending loan.
# Working customers can be targetted to lend loans as they have higher percentage of making payments on time.
# Customers with secondary education are most likely to make payments when compared to customers with academic degree.
# Married customers have paid loan amount on time when compared to widows.
# Customers owning House/apartment are most likely to make payments on time compared to those living in CO-OP apartment.
# Labourers have high repayement percentage. Hence baks can think of lending small amount loans to them.


# In[ ]:


# Correlation Matrix


# In[165]:


#correlation matrix for all numerical columns
corr=target_0.corr()
corr


# In[167]:


#correlation matrix for all numerical columns
corr1=target_1.corr()
corr1


# In[168]:


corr.style.background_gradient(cmap ='coolwarm')


# In[206]:


# Getting  top 10 correlation for the Repayers dataframe

corr_repayer = target_0.corr()
corr_df_repayer = corr_repayer.where(np.triu(np.ones(corr_repayer.shape),k=1).astype(np.bool)).unstack().reset_index()
corr_df_repayer.columns =['VAR1','VAR2','Correlation']
corr_df_repayer.dropna(subset = ["Correlation"], inplace = True)
corr_df_repayer["Correlation"]=corr_df_repayer["Correlation"].abs() 
corr_df_repayer.sort_values(by='Correlation', ascending=False, inplace=True) 
corr_df_repayer.head(10)


# In[205]:


#plotting heatmap to see linear correlation amoung Repayers 

fig = plt.figure(figsize=(40,15))
ax = sns.heatmap(target_0.corr(), cmap="RdYlGn",annot=True,linewidth =1)


# In[ ]:


# To get rid of the repeated correlation values between two variables we perform the following steps


# In[ ]:





# In[169]:


#Convert the diagonal and below diagonal values of matrix to False, Whereever False is there is replaced with NaN on execution
corr=corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
corr


# In[ ]:


#  Bivariate Analysis for target 0 and target 1


# In[ ]:


# Income vs Credit, Goods price vs Credit


# In[170]:


#Scatter plot for numeric columns

plt.figure(figsize = (15, 20))
plt.subplots_adjust(wspace=0.3)


plt.subplot(2,2,1)
sns.scatterplot(target_0.AMT_INCOME_TOTAL,target_0.AMT_CREDIT)
plt.xlabel('AMT_INCOME_TOTAL')
plt.ylabel('AMT_CREDIT')
plt.title('AMT_INCOME_TOTAL  vs  AMT_CREDIT ')

plt.subplot(2,2,2)
sns.scatterplot(target_1.AMT_INCOME_TOTAL,target_1.AMT_CREDIT)
plt.xlabel('AMT_INCOME_TOTAL')
plt.ylabel('AMT_CREDIT')
plt.title('AMT_INCOME_TOTAL  vs  AMT_CREDIT ')

plt.subplot(2,2,3)
sns.scatterplot(target_0.AMT_GOODS_PRICE,target_0.AMT_CREDIT)
plt.xlabel('AMT_GOODS_PRICE')
plt.ylabel('AMT_CREDIT')
plt.title('AMT_GOODS_PRICE  vs  AMT_CREDIT ')
plt.xticks(rotation = 45)

plt.subplot(2,2,4)
sns.scatterplot(target_1.AMT_GOODS_PRICE,target_1.AMT_CREDIT)
plt.xlabel('AMT_GOODS_PRICE')
plt.ylabel('AMT_CREDIT')
plt.title('AMT_GOODS_PRICE  vs  AMT_CREDIT ')
plt.xticks(rotation = 45)
plt.show()


# In[ ]:


# hose who have paid the loan amount on/within time are more likely to get higher credits than those who didnt pay/did late
payments.People who have higher goods price and have made payments on time have higher credits than those with higher goods 
price but didnt pay loan.


# In[ ]:


# Numerical categorical analysis


# In[ ]:


# Income range- Gender


# In[171]:


# Numeric variable analysis for target_0 & target_1 dataframe
plt.figure(figsize = (15, 8))
plt.subplot(2, 2, 1)
plt.title('Target_0:Income Range b/w Male and Female')

sns.countplot(x='AMT_INCOME_RANGE', hue='CODE_GENDER', data=target_0, palette='rocket')

# subplot 2
plt.subplot(2, 2, 2)
plt.title('Target_1:Income Range b/w Male and Female')

sns.countplot(x='AMT_INCOME_RANGE', hue='CODE_GENDER', data=target_1,palette='rocket')
plt.show()


# In[ ]:


# We can see that Females with low income don’t have any payment issues.


# In[ ]:


# Credit amount vs Education Status


# In[172]:


# Box plotting for Credit amount

plt.figure(figsize=(15,10))
plt.subplots_adjust(wspace=0.3)

plt.subplot(121)
sns.boxplot(data =target_0, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v]')
plt.title('Credit amount vs Education Status')
plt.xticks(rotation=45)

plt.subplot(122)
sns.boxplot(data =target_1, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v]')
plt.title('Credit amount vs Education Status')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# From the above plot,we can see that
1.Some of the highly educated, married person are having credits higher than those who have done lower secondary eduction.
2.Those with higher eduction have higher credits and are more likely to make payments on time.
3.More number of outliers are seen in higher education.
4.The people with secondary and secndary special eduction are less likely to make payments on time.


# In[ ]:


# Income vs Education Status


# In[173]:


# Box plotting for Income amount in logarithmic scale

plt.figure(figsize=(18,15))
plt.subplot(1,2,1)
plt.yscale('log')
sns.boxplot(data =target_0, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income amount vs Education Status(Target 0)')
plt.xticks(rotation=45)

plt.subplot(1,2,2)
plt.yscale('log')
sns.boxplot(data =target_1, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income amount vs Education Status (Target 1)')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# From the above plots,
1.we can see that Higher education has many outliers.
2.People with higher eductaion have higher income and dont have difficulties in making loan payment.
3.People with higher education who ave lesser income are unable to pay the loan.
Hence we can conclude that,people with Higher income are most likely to make payments.


# In[ ]:


# Reading the previous application


# In[174]:


df1.head()


# In[175]:


# Removing the column values of 'XNA' and 'XAP'

df1=df1.drop(df1[df1['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
df1=df1.drop(df1[df1['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
df1=df1.drop(df1[df1['NAME_CASH_LOAN_PURPOSE']=='XAP'].index)


# In[181]:


#Merge the previous application with the current application data file
merged_df2= pd.merge(df2, df1, how='inner', on='SK_ID_CURR',suffixes='_x')
merged_df2.head()


# In[182]:


merged_df2.columns


# In[183]:


merged_df2.head()


# In[185]:


# Renaming the column names after merging

new_df = merged_df2.rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY',
                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',
                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START','NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV',
                         'AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV',
                         'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',
                         'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'}, axis=1)
new_df.head()


# In[186]:


# Removing unwanted columns for analysis

new_df.drop(['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 
              'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
              'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY','WEEKDAY_APPR_PROCESS_START_PREV',
              'HOUR_APPR_PROCESS_START_PREV', 'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)

new_df.head()


# In[187]:


new_df.head()


# In[188]:


# Univariate Analysis


# In[197]:


# Distribution of contract status in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(20,20))

plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of contract status with purposes')
ax=sns.countplot(data = new_df, y='NAME_CASH_LOAN_PURPOSE', order=new_df['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS',palette='deep')


# In[ ]:


# Points to be concluded from above plot:
Most rejection of loans came from purpose 'Repairs'. For education purposes we have equal number of approves and rejection 
PayinG other loans and buying a new car is having significant higher rejection than approves.


# In[199]:


# Distribution of contract status

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(20,20))

plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of purposes with target ')
ax = sns.countplot(data = new_df, y= 'NAME_CASH_LOAN_PURPOSE', order=new_df['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'TARGET')


# In[ ]:


Few points we can conclude from above plot:
Loan purposes with 'Repairs' are facing more difficulites in payment on time. There are few places where loan payment is 
significant higher than facing difficulties. They are 'Buying a garage', 'Business developemt', 'Buying land','Buying a new car'
and 'Education' Hence we can focus on these purposes for which the client is having for minimal payment difficulties


# In[ ]:


# Bivariate Analysis


# In[ ]:


# Prev Credit amount vs Loan Purpose


# In[202]:


# Box plotting for Credit amount in logarithmic scale

plt.figure(figsize=(30,15))
plt.xticks(rotation=90)
plt.yscale('log')

sns.boxplot(data =new_df, x='NAME_CASH_LOAN_PURPOSE',hue='NAME_INCOME_TYPE',y='AMT_CREDIT_PREV',orient='v')
plt.title('Prev Credit amount vs Loan Purpose')
plt.show()


# In[ ]:


#From the above we can conclude some points- The credit amount of Loan purposes like 'Buying a home','Buying a land','Buying 
a new car' and'Building a house' is higher. Income type of state servants have a significant amount of credit applied Money for 
third person or a Hobby is having less credits applied for.


# In[203]:


# Box plotting for Credit amount prev vs Housing type in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
sns.barplot(data =new_df, y='AMT_CREDIT_PREV',hue='TARGET',x='NAME_HOUSING_TYPE',)
plt.title('Prev Credit amount vs Housing type')
plt.show()


# In[ ]:


# Here for Housing type, office appartment is having higher credit of target 0 and co-op apartment is having higher credit of 
target 1. So, we can conclude that bank should avoid giving loans to the housing type of co-op apartment as they are having 
difficulties in payment. Bank can focus mostly on housing type with parents or House\appartment or miuncipal appartment for 
successful payments.


# In[ ]:


CONCLUSION :
1.	Banks should focus more on Age group 30-50 as they have highest no. of payment issues…….
2.	Banks should focus more on income type ‘Working’ as they are having most number of  payment issue.
3.	Banks should focus more on MEDIUM income group people and  as they again have payment issue
4.	Customers owning House/apartment are most likely to make payments on time compared to those living in CO-OP apartment.
5.	Get as much as clients from housing type ‘With parents’ as they are having least number of unsuccessful payments.


# In[ ]:




