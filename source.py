#!/usr/bin/env python
# coding: utf-8

# # AI Developer Productivity, Task Success, and Emerging Technology Trends📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# As we observe, AI has changed the way developers build software from writing line by line to architect complex solutions, they can work faster and more efficiently. With the way things are going, lots of ideas and creative technologies have been created. Nowadays, developers can use tools like AI models to help generate code, code review, debug errors, create tests, and even architecture design. 
# 
# The problem I tried to study is to find out if increased use of AI tools is actually associated with higher developers productivity or if it brings more issues related to additional debugging, verification, and rework.
# 
# This project aims to use metrics such as coding time, AI usage, bugs reported, and task success to examine developer productivity and its relationship with AI development tools. In addition, I will look at the model families developers report using and compare coding benchmark scores across selected model versions.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# I'd like to ask how AI usage is changing software developement in term of developer productivity, developer experience, and technology trends.
# The analysis will:
# - Analyze how AI usage is related to implementation time, post-review work, perceived effort, and task completion.
# 
# - Examine how often developers use AI tools, how trust differs across usage groups, what frustrations current AI users report, and which model families they have used.
# 
# - Compare Coding Index scores across selected model versions and discuss these results alongside the model families reported in the survey.

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# My hypothesis is that the increased use of AI development tools will generally be associated with higher productivity and faster task completion. Along with that, I expect the increased use of AI may introduce additional debugging, verification, or code-quality problems.
# 
# Therefore, I think the results will show that AI can improve productivity while also bringing tradeoffs.
# 
# Some answers may be:
# - Developers with AI usage may complete the initial implementation faster.
# 
# - AI usage may affect the amount of time spent making changes after code review.
# 
# - Developers who use AI tools more frequently may report higher trust in AI output.
# 
# - Higher AI usage may also introduce additional debugging or quality-related concerns.
# - Selected model versions may have different coding benchmark scores, even within the same model family.
# 
# This project aims to study about the correlation and relationships, rather than claiming that AI directly changes developer productivity. 

# ## Data Sources
# 
# This project will use three data sources:
# 
# - METR Developer Productivity Study (CSV) will be used to collect developer productivity which includes tasks completed with and without AI assistance.
# 
# - Stack Overflow Developer Survey 2025 (CSV) will be used to examine developers' AI usage, trust, frustrations, and the model families they have used. 
# 
# - Artificial Analysis Data API (API) will be used to collect model names, creators, release dates, and Coding Index scores.
# 
# The datasets will not be directly merged but instead, their results will be used to compare in the final analysis to provide different perspectives on how AI is affecting software development.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# - METR Developer Productivity Study: this data source provides a list of developers with AI and non-AI usage. I can use metrics like `implementation time`, `post-review time`, `task completion`, and `perceived effort` to compare productivity between these two groups. Bar charts may be used to support the analysis through visualization.
# 
# - Stack Overflow Developer Survey 2025: I will use `AISelect`, `AIAcc`, and `AIFrustration` to examine AI usage frequency, compare trust across usage groups, and identify common frustrations among current AI users. I will also use `AIModelsHaveWorkedWith` to identify the model families developers have used.
# 
# - Artificial Analysis API: I will use the Coding Index to compare coding benchmark scores across selected model versions. A bar chart can show how their scores differ. I will discuss these results alongside the survey model usage responses. The survey lists model families, while the API scores specific versions, so I will keep the two results separate.

# # Setup and Import Libraries
# 

# In[1]:


get_ipython().system('pip install python-dotenv')


# In[2]:


# Import libraries
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import requests


# # Data Import and Initial Inspection
# 
# 

# ### Import METR Data - Open Source Developer Productivity Data

# In[3]:


metr_df = pd.read_csv("data/raw/osdp.csv")
metr_df.shape


# In[4]:


metr_df.head()


# In[5]:


metr_df.info()


# ### Import Stack Overflow Data
# 
# Because the original dataset is over 100MB, I couldn't push it to GitHub without setting up GitHub LFS, please download the data source from [Stack Overflow Developer Survey 2025](tps://github.com/StackExchange/Survey/blob/main/packages/archive/2025/results.csv) and rename it to `stackoverflowdata.csv` and move it to `data/raw`

# In[6]:


stackoverflow_df = pd.read_csv("data/raw/stackoverflowdata.csv")
stackoverflow_df.shape


# In[7]:


stackoverflow_df.head()


# In[8]:


stackoverflow_df.info()


# ### Import Data From Artificial Analysis API
# 

# For this part, I will use the [Artificial Analysis Data API](https://artificialanalysis.ai/data-api/docs) to collect model names, creators, release dates, and Coding Index scores. The free endpoint returns JSON.
# 
# I use `ARTIFICIAL_ANALYSIS_API_KEY` set in .env file. The code will read all pages and keep the retrieval date in `model_scores_df`.
# 
# The Coding Index describes performance on coding benchmarks. It does not measure the time a developer spends coding.
# 

# In[9]:


# Import model benchmark data from Artificial Analysis.
load_dotenv(".env")
aa_api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY", "")
if not aa_api_key:
    raise ValueError("An Artificial Analysis API key is required.")
url = "https://artificialanalysis.ai/api/v2/language/models/free"

model_records = []
page = 1

# Fetch all pages of model benchmark data
while True:
    response = requests.get(
        url,
        headers={"x-api-key": aa_api_key},
        params={"page": page},
        timeout=30
    )
    response.raise_for_status()
    data = response.json()
    model_records.extend(data["data"])

    if not data["pagination"]["has_more"]:
        break
    page += 1

model_scores_df = pd.json_normalize(model_records)
model_scores_df["retrieved_at"] = pd.Timestamp.now(tz="UTC")
model_scores_df.to_json(
    "data/raw/artificial_analysis_models.json",
    orient="records",
    date_format="iso"
)
model_scores_df.head()


# # Data Cleaning and Preparation
# 
# For this part, I will prepare each dataset separately. I will check for missing values and duplicate records, then decide how to handle them based on the analysis.
# 

# ### Prepare METR Data
# 
# For METR, I will first check how many values are missing in each column.
# 

# In[ ]:


metr_df.isnull().sum()


# Four columns have missing values: `initialImplementationTime`, `postReviewImplementationTime`, `estimatedTimeOtherTreatment`, and `perceivedEffort`.
# 
# Next, I will check which task statuses have these missing values. This will help me decide which tasks to use for each comparison.
# 

# In[ ]:


metr_df["status"].value_counts()


# In[ ]:


missing_fields = metr_df.columns[metr_df.isnull().any()]

# Filter to only includes rows with missing values
metr_missing_by_status = metr_df.groupby('status')[missing_fields].apply(lambda x: x.isnull().sum())
metr_missing_by_status['task_count'] = metr_df.groupby('status').size()
metr_missing_by_status


# All 828 completed tasks have values in these four columns. The missing values only appear in incomplete or ditched tasks.
# 
# Next, I will check for duplicate task IDs.
# 

# In[ ]:


metr_df["id"].duplicated().sum()


# There are no duplicate task IDs, so no records need to be removed at this step.
# 
# I will keep 3 statuses (complete, incomplete, and ditched tasks) for the task completion analysis and exclude 5 records labeled cheating or corrupted. For time and effort comparisons, I will use completed tasks. I will keep the original DataFrame and will not fill missing values with 0.

# In[ ]:


# Keep the three status for the status comparison
metr_status_df = metr_df[metr_df['status'].isin(['complete','ditched','incomplete'])].copy()

# Keep completed tasks for time and effort comparison
metr_complete_df = metr_status_df[metr_status_df['status'] == 'complete'].copy()


# ### Prepare Stack Overflow Data
# 

# I will keep `ResponseId` and `AIModelsHaveWorkedWith` in `stackoverflow_models_df` for the model usage question. I will count the available model answers later in the EDA.
# 

# In[ ]:


stackoverflow_models_df = stackoverflow_df[
    ["ResponseId", "AIModelsHaveWorkedWith"]
].copy()
model_counts = (
    stackoverflow_models_df["AIModelsHaveWorkedWith"]
    .str.split(";")
    .explode()
    .value_counts()
)

model_counts


# I will keep the 10 columns selected for the project. The usage, trust, and frustration questions will focus on `AISelect`, `AIAcc`, and `AIFrustration`. `AISent` and `AIComplex` will be used to answer further questions on people who use AI frequently and somewhat trust it to see how they think about it
# 

# In[ ]:


# Keep the columns used in the analysis.
col_used = [
    "Employment",
    "DevType",
    "WorkExp",
    "YearsCode",
    "LanguageHaveWorkedWith",
    "AISelect",
    "AISent",
    "AIAcc",
    "AIComplex",
    "AIFrustration"
]

stackoverflow_df = stackoverflow_df[col_used].copy()
stackoverflow_df.head()


# Next, I will check which of the selected columns have missing values.
# 

# In[ ]:


stackoverflow_df.isna().sum()


# I will remove rows with missing values across the 5 columns `AISelect`, `AISent`, `AIAcc`, `AIComplex`, and `AIFrustration` and report the number of rows before and after. 
# 
# 

# In[ ]:


rows_before = len(stackoverflow_df)
stackoverflow_df = stackoverflow_df.dropna().copy()
rows_after = len(stackoverflow_df)
print(f"Rows before dropping NaN: {rows_before}")
print(f"Rows after dropping NaN: {rows_after}")


# About 45% of the records were dropped, because they had missing values in the selected columns. This will leave complete records for the analysis, but it may also introduce bias.

# ### Prepare Artificial Analysis Data
# 
# For Artificial Analysis data, I will keep the model `name`, `creator`, `release_date`, `coding_index`, and `retrieval_date`. I will rename `evaluations.artificial_analysis_coding_index` to `coding_index` and convert the scores to numeric values. Models without a score will be left out of the benchmark comparison.
# 

# In[ ]:


col_used = ["name",
            "model_creator.name",
            "evaluations.artificial_analysis_coding_index",
            "release_date",
            "retrieved_at"
        ]

model_scores_df = model_scores_df[col_used].copy()
model_scores_df.head()


# 
# 
# I will keep the original DataFrame and create `model_scores_clean_df` in the next cell.

# In[ ]:





# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# [Open Source Developer Productivity - Late 2025 Follow-on Study Results ](https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs)
# 
# [Stack Overflow Developer Survey](https://github.com/StackExchange/Survey)
# 
# [Artificial Analysis](https://artificialanalysis.ai/)
# 
# [Artificial Analysis Data API documentation](https://artificialanalysis.ai/data-api/docs)

# ## AI Use Disclosure
# 
# I used AI tools to refine project scope and data analysis ideas, improve written explanations. I also used AI to help compare my project scope with the available datasets to ensure the data I selected could actually help answer the questions. I reviewed the final analysis, interpretations, and conclusions myself.

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

