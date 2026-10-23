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

# In[91]:


get_ipython().system('pip install python-dotenv matplotlib')


# In[92]:


# Import libraries
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px


# # Data Import
# 
# 

# ### Import METR Data 

# In[93]:


metr_df = pd.read_csv("data/raw/osdp.csv")
metr_df.shape


# In[94]:


metr_df.head()


# In[95]:


metr_df.info()


# ### Import Stack Overflow Data
# 
# Because the original dataset is over 100MB, I couldn't push it to GitHub without setting up GitHub LFS, please download the data source from [Stack Overflow Developer Survey 2025](tps://github.com/StackExchange/Survey/blob/main/packages/archive/2025/results.csv) and rename it to `stackoverflowdata.csv` and move it to `data/raw`

# In[96]:


stackoverflow_df = pd.read_csv("data/raw/stackoverflowdata.csv")
stackoverflow_df.shape


# In[97]:


stackoverflow_df.head()


# In[98]:


stackoverflow_df.info()


# ### Import Data From Artificial Analysis API
# 

# For this part, I will use the [Artificial Analysis Data API](https://artificialanalysis.ai/data-api/docs) to collect model names, creators, release dates, and Coding Index scores. The free endpoint returns JSON.
# 
# I use `ARTIFICIAL_ANALYSIS_API_KEY` set in .env file. The code will read all pages and keep the retrieval date in `model_scores_df`.
# 
# The Coding Index describes performance on coding benchmarks. It does not measure the time a developer spends coding.
# 

# In[99]:


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


# In[100]:


model_scores_df.info()


# # Data Cleaning and Initial EDA
# 
# For this part, I will prepare each dataset separately. I will check for missing values, duplicate records, and other abnormal values then decide how to handle them based on the analysis.
# 

# ### Prepare METR Data
# 
# At first, I want to see a overview of METR dataset by using describe() 

# In[101]:


metr_df.describe()


# I see some abnormal values on time related columns. For example, `initialImplementationTime` can be right-skewed because its mean value is 159.29, but median is 87, and max is 1685. Mean is much higher than median, there may be some long time tasks making mean value higher. But I'll first check missing values and duplicated values and then will check the describe statistical again.

# In[102]:


metr_df.isnull().sum()


# Four columns have missing values: `initialImplementationTime`, `postReviewImplementationTime`, `estimatedTimeOtherTreatment`, and `perceivedEffort`.
# 
# Next, I will check which task statuses have these missing values. This will help me decide which tasks to use for each comparison.
# 

# In[103]:


metr_df["status"].value_counts()


# In[104]:


missing_fields = metr_df.columns[metr_df.isnull().any()]

# Filter to only includes rows with missing values
metr_missing_by_status = metr_df.groupby('status')[missing_fields].apply(lambda x: x.isnull().sum())
metr_missing_by_status['task_count'] = metr_df.groupby('status').size()
metr_missing_by_status


# All 828 completed tasks have values in these four columns. The missing values only appear in incomplete or ditched tasks.
# 
# Next, I will check for duplicate task IDs.
# 

# In[105]:


metr_df["id"].duplicated().sum()


# There are no duplicate task IDs, so no records need to be removed at this step.
# 
# I will keep 3 statuses (complete, incomplete, and ditched tasks) for the task completion analysis and exclude 5 records labeled cheating or corrupted. For time and effort comparisons, I will use completed tasks, any analysis established for this dataframe will not represent the rest of the dataset. I will keep the original DataFrame and will not fill missing values with 0.

# In[106]:


# Keep the three status for the status comparison
metr_status_df = metr_df[metr_df['status'].isin(['complete','ditched','incomplete'])].copy()

# Keep completed tasks for time and effort comparison
metr_complete_df = metr_status_df[metr_status_df['status'] == 'complete'].copy()


# In[107]:


metr_complete_df.describe()


# These values are still a little abnormal, so I will sketch a histogram to check the distribution of `initialImplementationTime`

# In[108]:


fig = plt.figure(figsize=(10, 6))
plt.hist(metr_complete_df["initialImplementationTime"], bins=30, edgecolor='black')
plt.xlabel("Initial Implementation Time (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Initial Implementation Time for Completed Tasks")

plt.show()


# From this chart, I can say the `initialImplementationTime` is right-skewed when most values concentrates between 0-500, while some tasks have implementation time up to 1700 minutes, it creates a long right tail.
# 
# There are also some potential outliers for time above 1000, which may affect `initialImplementationTime` but they may not be removed from the dataframe since some complex tasks may take longer time to complete.

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


# In[122]:


stackoverflow_models_df.duplicated().sum()


# No duplicate ResponseId values were found, so no responses were removed at this step.

# I will keep the 10 columns selected for the project. The usage, trust, and frustration questions will focus on `AISelect`, `AIAcc`, and `AIFrustration`. `AISent` and `AIComplex` will be used to answer further questions on people who use AI frequently and somewhat trust it to see how they think about it
# 

# In[110]:


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

# In[111]:


stackoverflow_df.isna().sum()


# I will remove rows with missing values across the 5 columns `AISelect`, `AISent`, `AIAcc`, `AIComplex`, and `AIFrustration`. Along with that, I will split into 3 different dataframe for each analysis, that way I can preverse as many values as possble for later analysis.
# 
# 

# In[119]:


# Keep available responses for AI usage analysis.
stackoverflow_usage_df = stackoverflow_df[["AISelect"]].dropna().copy()

# Keep responses to compare trust across usage groups.
stackoverflow_trust_df = stackoverflow_df[
    ["AISelect", "AIAcc"]
].dropna().copy()

# Keep responses for frustration analysis.
stackoverflow_frustration_df = stackoverflow_df[
    ["AISelect", "AIFrustration"]
].dropna().copy()


# In[121]:


stackoverflow_usage_df.value_counts()


# ### Prepare Artificial Analysis Data
# 
# For Artificial Analysis data, I will keep the model `name`, `creator`, `release_date`, `coding_index`, and `retrieval_date`. I will rename `evaluations.artificial_analysis_coding_index` to `coding_index` and models without a score will be left out of the benchmark comparison.
# 

# In[113]:


col_used = ["name",
            "model_creator.name",
            "evaluations.artificial_analysis_coding_index",
            "release_date",
            "retrieved_at"
        ]

model_scores_df = model_scores_df[col_used].copy()
model_scores_df.head()


# Next, I will rename column to `coding_index` and remove missing values from `coding_index`.

# In[114]:


model_scores_df= model_scores_df.rename(columns={"evaluations.artificial_analysis_coding_index": "coding_index"})


# In[202]:


model_scores_df.isna().sum()


# In[203]:


model_scores_df = model_scores_df.dropna().copy()


# I will check duplicated values

# In[116]:


model_scores_df.duplicated().sum()


# Since this benchmark comparison is based on the 2025 survey data, I will only include AI models with a release_date on or before December 31, 2025.

# In[117]:


model_scores_df = model_scores_df[model_scores_df['release_date']<="2025-12-31"].copy()
model_scores_df.info()


# In[171]:


model_scores_df.head(10)


# To keep the comparison focused, I keep the five most reported model families. For each family, I will then keep up to five benchmark versions with the highest Coding Index scores.

# In[ ]:


# Map model name to their family

survey_mapping = {
    "openAI GPT (chatbot models)": "OpenAI",
    "openAI Reasoning models": "OpenAI",
    "Gemini (Flash general purpose models)": "Gemini",
    "Gemini (Pro Reasoning models)": "Gemini",
    "DeepSeek (R- Reasoning models)": "DeepSeek",
    "DeepSeek (V- General purpose models)": "DeepSeek",
    "Anthropic: Claude Sonnet": "Claude Sonnet",
    "Meta Llama (all models)": "Meta Llama"
}

# Split the AIModelsHaveWorkedWith column into a list of model families
survey_model_df = stackoverflow_models_df.copy()
survey_model_df["model_family"] = (
    survey_model_df["AIModelsHaveWorkedWith"].str.split(";")
)

# explode the model_family column to have one model per row and drop rows with missing values
survey_model_df = (survey_model_df.explode("model_family").dropna(subset=["model_family"]))

# Filter out the openAI Image generating models as they are not relevant to the analysis
survey_model_df = survey_model_df[survey_model_df["model_family"] != "openAI Image generating models"].copy()

# Map the model families to their corresponding family names using the survey_mapping dictionary
survey_model_df["model_family"] = (survey_model_df["model_family"].replace(survey_mapping))


top5_families = survey_model_df["model_family"].value_counts().head(5)
top5_families


# Here, I found the five model families most often reported by Stack Overflow survey respondents. The Stack Overflow survey shows which model families developers have used most: OpenAI (19,140 mentions), Gemini (10,044), Claude Sonnet (7,063), DeepSeek (6,211), and Meta Llama (2,941).
# 
# I removed OpenAI's image generation models because they are not used for coding. Next, I will use these five families to look up the Coding Index scores of their model versions from Artificial Analysis.

# In[308]:


benchmark_family_df = model_scores_df.copy()

# Group models using provider name
benchmark_family_df["model_family"] = (benchmark_family_df["model_creator.name"].map({
    "OpenAI": "OpenAI",
    "DeepSeek": "DeepSeek",
    "Meta": "Meta Llama"
}))

# Group Gemini Flash and Pro versions together
benchmark_family_df.loc[benchmark_family_df["name"].str.startswith("Gemini") 
                        & benchmark_family_df["name"].str.contains("Flash|Pro"),
                        "model_family"] = "Gemini"

# The survey only includes Claude Sonnet.
benchmark_family_df.loc[benchmark_family_df["name"].str.contains("Sonnet")
    & benchmark_family_df["model_creator.name"].eq("Anthropic"),
    "model_family"] = "Claude Sonnet"

# Choose model release in 2025
benchmark_family_df = benchmark_family_df[benchmark_family_df['release_date'].between('2025-01-01', '2025-12-31')].copy()

benchmark_family_df = benchmark_family_df.dropna(subset=["model_family"]).copy()

benchmark_family_df[["name", "model_family", "coding_index", "release_date"]].sample(5)


# In[318]:


# Keep 2 highest score model per model family
top_benchmark_df = (benchmark_family_df.sort_values("coding_index", ascending=False)
                     .groupby("model_family")
                     .head(2))
top_benchmark_df[["name", "model_family", "coding_index"]]


# # Exploratory Data Analysis

# In this section, I'll start exploring the chosen dataset including 

# ## Stack Overflow Data
# 

# ### Stack Overflow: AI Usage Frequency
# In this section, I will examine how frequently developers use AI tools by counting the responses in each `AISelect` category in `stackoverflow_usage_df`.

# In[134]:


AI_counts = stackoverflow_usage_df["AISelect"].value_counts()
fig = px.bar(AI_counts, x=AI_counts.index, y=AI_counts.values, labels={'x':'AI Usage', 'y':'Count'}, title='AI Usage Distribution')

# Add percentage of each category
fig.update_traces(
    customdata=(AI_counts.values / AI_counts.values.sum() * 100).round(2),
    hovertemplate='%{x}: %{y} (%{customdata}%)'
)
fig.show()


# About 15883 individuals use AI tools daily accounts for 47.1% of the total. Overall, 78,5% of those who answered this question reported using AI tools daily, weekly or infrequently. About 22% which is a small amount of responders saying they either plan to use AI tools in the future or don't even plan to use it.

# ### Stack Overflow
# At first, I'd like to see the selection and the amount of answers in `AIAcc` column.

# In[138]:


stackoverflow_trust_df["AIAcc"].value_counts()


# Then, I'd like to see the amount of people who use AI and their trust on those.

# In[316]:


#trust_counts = pd.crosstab(stackoverflow_trust_df["AISelect"], 
#                          stackoverflow_trust_df["AIAcc"])

# Trust percentages
trust_percent = pd.crosstab(
    stackoverflow_trust_df["AISelect"],
    stackoverflow_trust_df["AIAcc"],
    normalize="index"
) * 100

# Reindex AI usage categories
usage_order = [
    "No, and I don't plan to",
    "No, but I plan to soon",
    "Yes, I use AI tools monthly or infrequently",
    "Yes, I use AI tools weekly",
    "Yes, I use AI tools daily"
]
trust_percent = trust_percent.reindex(usage_order)

trust_percent = trust_percent.round(1)

trust_percent


# In[169]:


fig = px.bar(
    trust_percent,
    x=trust_percent.index,
    y=trust_percent.columns,
    title='Trust in AI Accuracy by AI Usage')

fig.update_layout(
    xaxis_title='AI Usage',
    yaxis_title='Percentage')

fig.show()


# This results suggest that among respondents who don't use AI tools and don't plan to, 60.7% **highly distrust** AI. In contrast, among daily AI users, 44.0% **somewhat trust** AI and 5.4% **highly trust** it. This suggests an association between AI usage frequency and trust, but it doesn't suggest causation.

# ### Artificial Analysis
# In this section, I will examine the coding index of the most 5 models used reported from Stack Overflow data, and comparing their score using Artificial Analysis data.

# In[346]:


fig = px.bar(top_benchmark_df, x="model_family", y="coding_index", color="name",
             barmode="group",
             title="Top 2 Highest Scoring Models per Model Family")

fig.update_layout(
    width=800,
    height=400,
    bargap=0.2,
    bargroupgap=0,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.3,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
        title_text=""
    ),
    margin=dict(l=60, r=40, t=80, b=160),
    xaxis_title="Model Family",
    yaxis_title="Coding Index"
)

fig.show()


# In[347]:


top5_families


# The Stack Overflow survey shows which model families developers have used most: OpenAI (19,140 mentions), Gemini (10,044), Claude Sonnet (7,063), DeepSeek (6,211), and Meta Llama (2,941).
# 
# From this chart, I can say that usage and Coding Index score are not clearly associated. OpenAI is the most reported family, but its best model (GPT-5.1 High, 49.4) scores below Claude 4.5 Sonnet (52.1), which ranks third in usage. DeepSeek is fourth in usage, but its two models score close to OpenAI and Gemini (44.2 and 43.5). Meta Llama is the only family that is last in both usage and score.
# 
# Therefore, this chart doesn't suggest that developers choose a model because of its Coding Index, or that a higher score leads to higher productivity.
# 

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# [Open Source Developer Productivity](https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs)
# 
# [Stack Overflow Developer Survey](https://github.com/StackExchange/Survey)
# 
# [Artificial Analysis](https://artificialanalysis.ai/)
# 
# [Artificial Analysis Data API documentation](https://artificialanalysis.ai/data-api/docs)
# 
# [Plotly](https://plotly.com/python/)

# ## AI Use Disclosure
# 
# I used AI tools to refine project scope and data analysis ideas, improve written explanations. I also used AI to help compare my project scope with the available datasets to ensure the data I selected could actually help answer the questions. I reviewed the final analysis, interpretations, and conclusions myself.

# In[118]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

