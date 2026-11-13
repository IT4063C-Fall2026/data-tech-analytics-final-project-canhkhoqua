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

# In[100]:


get_ipython().system('pip install python-dotenv matplotlib seaborn scikit-learn')


# In[101]:


# Import libraries
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# # Data Import
# 
# 

# ### Import METR Data 

# In[102]:


metr_df = pd.read_csv("data/raw/osdp.csv")
metr_df.shape


# In[103]:


metr_df.head()


# In[104]:


metr_df.info()


# ### Import Stack Overflow Data
# 
# Because the original dataset is over 100MB, I couldn't push it to GitHub without setting up GitHub LFS, please download the data source from [Stack Overflow Developer Survey 2025](tps://github.com/StackExchange/Survey/blob/main/packages/archive/2025/results.csv) and rename it to `stackoverflowdata.csv` and move it to `data/raw`

# In[105]:


stackoverflow_df = pd.read_csv("data/raw/stackoverflowdata.csv")
stackoverflow_df.shape


# In[106]:


stackoverflow_df.head()


# In[107]:


stackoverflow_df.info()


# ### Import Data From Artificial Analysis API
# 

# For this part, I will use the [Artificial Analysis Data API](https://artificialanalysis.ai/data-api/docs) to collect model names, creators, release dates, and Coding Index scores. The free endpoint returns JSON.
# 
# I use `ARTIFICIAL_ANALYSIS_API_KEY` set in .env file. The code will read all pages and keep the retrieval date in `model_scores_df`.
# 
# The API response is saved to `data/raw/artificial_analysis_models.json`. If this file exists, the code loads it instead of calling the API again. I use the snapshot retrieved on October 20, 2026, because the free endpoint later stopped returning Coding Index scores, so a new API call would leave no models to compare.
# 
# The Coding Index describes performance on coding benchmarks. It does not measure the time a developer spends coding.
# 

# In[108]:


# Import model benchmark data from Artificial Analysis.
aa_path = "data/raw/artificial_analysis_models.json"

if os.path.exists(aa_path):
    # Use the saved snapshot so results stay the same across runs
    model_scores_df = pd.read_json(aa_path, orient="records", dtype=False)
else:
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
        aa_path,
        orient="records",
        date_format="iso"
    )

model_scores_df.head()


# In[109]:


model_scores_df.info()


# # Data Cleaning and Initial EDA
# 
# For this part, I will prepare each dataset separately. I will check for missing values, duplicate records, and other abnormal values then decide how to handle them based on the analysis.
# 

# ### Prepare METR Data
# 
# At first, I want to see a overview of METR dataset by using describe() 

# In[110]:


metr_df.describe()


# I see some abnormal values on time related columns. For example, `initialImplementationTime` can be right-skewed because its mean value is 159.29, but median is 87, and max is 1685. Mean is much higher than median, there may be some long time tasks making mean value higher. But I'll first check missing values and duplicated values and then will check the describe statistical again.

# In[111]:


metr_df.isnull().sum()


# Four columns have missing values: `initialImplementationTime`, `postReviewImplementationTime`, `estimatedTimeOtherTreatment`, and `perceivedEffort`.
# 
# Next, I will check which task statuses have these missing values. This will help me decide which tasks to use for each comparison.
# 

# In[112]:


metr_df["status"].value_counts()


# In[113]:


missing_fields = metr_df.columns[metr_df.isnull().any()]

# Filter to only includes rows with missing values
metr_missing_by_status = metr_df.groupby('status')[missing_fields].apply(lambda x: x.isnull().sum())
metr_missing_by_status['task_count'] = metr_df.groupby('status').size()
metr_missing_by_status


# All 828 completed tasks have values in these four columns. The missing values only appear in incomplete or ditched tasks.
# 
# Next, I will check for duplicate task IDs.
# 

# In[114]:


metr_df["id"].duplicated().sum()


# There are no duplicate task IDs, so no records need to be removed at this step.
# 
# I will keep 3 statuses (complete, incomplete, and ditched tasks) for the task completion analysis and exclude 5 records labeled cheating or corrupted. For time and effort comparisons, I will use completed tasks, any analysis established for this dataframe will not represent the rest of the dataset. I will keep the original DataFrame and will not fill missing values with 0.

# In[115]:


# Keep the three status for the status comparison
metr_status_df = metr_df[metr_df['status'].isin(['complete','ditched','incomplete'])].copy()

# Keep completed tasks for time and effort comparison
metr_complete_df = metr_status_df[metr_status_df['status'] == 'complete'].copy()


# ### Check for data correlation in METR dataset

# In[116]:


sns.heatmap(metr_complete_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")


# In[117]:


metr_complete_df.describe()


# These values are still a little abnormal, so I will sketch a histogram to check the distribution of `initialImplementationTime`

# In[118]:


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

# In[119]:


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


# In[120]:


stackoverflow_models_df.duplicated().sum()


# No duplicate ResponseId values were found, so no responses were removed at this step.

# I will keep the 10 columns selected for the project. The usage, trust, and frustration questions will focus on `AISelect`, `AIAcc`, and `AIFrustration`.
# 

# In[121]:


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

# In[122]:


stackoverflow_df.isna().sum()


# I will remove rows with missing values across the 5 columns `AISelect`, `AISent`, `AIAcc`, `AIComplex`, and `AIFrustration`. Along with that, I will split into 3 different dataframe for each analysis, that way I can preverse as many values as possble for later analysis.
# 
# 

# In[123]:


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


# In[124]:


stackoverflow_usage_df.value_counts()


# ### Prepare Artificial Analysis Data
# 
# For Artificial Analysis data, I will keep the model `name`, `creator`, `release_date`, `coding_index`, and `retrieval_date`. I will rename `evaluations.artificial_analysis_coding_index` to `coding_index` and models without a score will be left out of the benchmark comparison.
# 

# In[125]:


col_used = ["name",
            "model_creator.name",
            "evaluations.artificial_analysis_coding_index",
            "release_date",
            "retrieved_at"
        ]

model_scores_df = model_scores_df[col_used].copy()
model_scores_df.head()


# Next, I will rename column to `coding_index` and remove missing values from `coding_index`.

# In[126]:


model_scores_df= model_scores_df.rename(columns={"evaluations.artificial_analysis_coding_index": "coding_index"})


# In[127]:


model_scores_df.isna().sum()


# In[128]:


model_scores_df = model_scores_df.dropna().copy()


# I will check duplicated values

# In[129]:


model_scores_df.duplicated().sum()


# Since this benchmark comparison is based on the 2025 survey data, I will only include AI models with a release_date on or before December 31, 2025.

# In[130]:


model_scores_df = model_scores_df[model_scores_df['release_date']<="2025-12-31"].copy()
model_scores_df.info()


# In[131]:


model_scores_df.head(10)


# To keep the comparison focused, I keep the five most reported model families. For each family, I will then keep up to five benchmark versions with the highest Coding Index scores.

# In[132]:


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

# In[133]:


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


# In[134]:


# Keep 2 highest score model per model family
top_benchmark_df = (benchmark_family_df.sort_values("coding_index", ascending=False)
                     .groupby("model_family")
                     .head(2))
top_benchmark_df[["name", "model_family", "coding_index"]]


# # Exploratory Data Analysis

# In this section, I'll start exploring the chosen dataset including 

# ## METR Data
# 
# ### METR: Productivity With and Without AI
# In this section, I will compare AI-allowed and AI-disallowed tasks. I use `metr_complete_df` for implementation time, post-review time, and perceived effort, and `metr_status_df` for the completion rate. Because the time columns are right-skewed on the above visulization, I will report the median implementation time instead of the mean.

# In[135]:


# Compare time and effort on completed tasks
metr_summary = metr_complete_df.groupby("treatmentAssignment").agg(
    completed_tasks=("id", "count"),
    median_implementation_time=("initialImplementationTime", "median"),
    mean_post_review_time=("postReviewImplementationTime", "mean"),
    mean_perceived_effort=("perceivedEffort", "mean")
)

# Completion rate uses complete, incomplete, and ditched tasks
metr_summary["completion_rate"] = (
    metr_status_df.groupby("treatmentAssignment")["status"]
    .apply(lambda x: (x == "complete").mean() * 100)
)

metr_summary.round(2)


# In[136]:


fig = px.box(metr_complete_df, x="treatmentAssignment", y="initialImplementationTime",
             log_y=True,
             labels={"treatmentAssignment": "Treatment",
                     "initialImplementationTime": "Initial Implementation Time (minutes, log scale)"},
             title="Initial Implementation Time by AI Treatment (Completed Tasks)")
fig.show()


# AI-allowed tasks have a higher completion rate (75.7%) than AI-disallowed tasks (70.6%).
# Findings:
# 
# - The median implementation time is 90 minutes for AI-allowed tasks and 82.5 minutes for AI-disallowed tasks.
# - The mean post-review time is a little higher for AI-allowed tasks (22.4 vs 17.9 minutes). 
# - The mean perceived effort is almost the same (2.62 vs 2.59).
# 
# Overall, AI-allowed tasks were completed slightly more often, but they were not clearly faster, did not need less post-review work, and did not feel easier.

# ## Stack Overflow Data
# 

# ### Stack Overflow: AI Usage Frequency
# In this section, I will examine how frequently developers use AI tools by counting the responses in each `AISelect` category in `stackoverflow_usage_df`.

# In[137]:


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

# In[139]:


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


# In[140]:


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

# ### Stack Overflow: AI Frustrations
# In this section, I will look at the frustrations reported by AI users. I keep respondents whose `AISelect` answer starts with "Yes", then split `AIFrustration` because each respondent can choose more than one answer.

# In[146]:


# Keep AI users only
AI_users_df = stackoverflow_frustration_df[
    stackoverflow_frustration_df["AISelect"].str.startswith("Yes")
]

frustration_counts = (
    AI_users_df["AIFrustration"].str.split(";").explode().value_counts()
)

# Percentage of AI users who selected each frustration
frustration_percent = (frustration_counts / len(AI_users_df) * 100).round(1)

fig = px.bar(frustration_percent.sort_values(), orientation="h",
             labels={"value": "Percentage of AI Users", "AIFrustration": "Frustration"},
             title="Frustrations Reported by AI Users")
fig.update_layout(showlegend=False)
fig.show()


# 24,827 AI users who answered this question, about three in four (74.9%) said AI gives answers that are "almost right, but not quite." Almost half (48.5%) said debugging AI-generated code takes more time. Only 4.5% said they haven't run into any problems.
# 
# This fits the second part of my hypothesis: using AI often means extra time spent checking and fixing its output. It could also help explain the METR results above. AI-allowed tasks weren't clearly faster there, and they took slightly more post-review time (22.4 vs 17.9 minutes on average).

# ### Artificial Analysis
# In this section, I will examine the coding index of the most 5 models used reported from Stack Overflow data, and comparing their score using Artificial Analysis data.

# In[142]:


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


# In[143]:


top5_families


# The Stack Overflow survey shows which model families developers have used most: OpenAI (19,140 mentions), Gemini (10,044), Claude Sonnet (7,063), DeepSeek (6,211), and Meta Llama (2,941).
# 
# From this chart, I can say that usage and Coding Index score are not clearly associated. OpenAI is the most reported family, but its best model (GPT-5.1 High, 49.4) scores below Claude 4.5 Sonnet (52.1), which ranks third in usage. DeepSeek is fourth in usage, but its two models score close to OpenAI and Gemini (44.2 and 43.5). Meta Llama is the only family that is last in both usage and score.
# 
# Therefore, this chart doesn't suggest that developers choose a model because of its Coding Index, or that a higher score leads to higher productivity.
# 

# ## Machine Learning Plan
# 
# Since I have 3 different datasets and I do not join them, I need to choose one dataset for the machine learning part. The Stack Overflow survey contains mostly multiple-choice text answers, and the Artificial Analysis data contains benchmark scores of AI models, not information about developers or tasks. The METR dataset is the only one that has one row per task with a measured outcome (time), so I will use METR for this part.
# 
# ### What type of machine learning model are you planning to use?
# 
# I want to answer this question: if we know AI usage, prior experience, and the developer's estimated task time, can we predict how long a developer takes to complete a task?
# 
# The target `initialImplementationTime` is a number (minutes), so this is a supervised **regression** problem. I plan to use **Linear Regression** because it is simple and its coefficients show how each feature (for example AI-allowed vs AI-disallowed) is related to completion time. I will also compare it with a baseline model that always predicts the median time, so I can check if the model actually learns something.
# 
# - **Data:** `metr_complete_df` (828 completed tasks)
# - **Target:** `initialImplementationTime`
# - **Features** (all known before the task starts):
#   - `treatmentAssignment` (AI-allowed / AI-disallowed)
#   - `priorExperience` (1-5)
#   - `estimatedTime`: the developer's estimate for the assigned condition (`estimatedTimeWithAI` for AI-allowed tasks, `estimatedTimeWithoutAI` for AI-disallowed tasks)
# 
# ### What are the challenges you have identified/are you anticipating in building your machine learning model?
# 
# 1. **Data leakage:** `estimatedTimeOtherTreatment`, `postReviewImplementationTime`, and `perceivedEffort` are recorded after the task is finished. `estimatedTimeOtherTreatment` has a 0.78 correlation with the target, so using it would make the model look much better than it really is.
# 2. **Skewed target and outliers:** from the EDA, `initialImplementationTime` is right-skewed (median 87 minutes, max 1685 minutes). A few very long tasks can pull the regression line and make the errors large.
# 3. **Same developers appear many times:** the 828 tasks come from only 53 developers (from 1 to 75 tasks each). If I split the data randomly, the same developer can be in both the training and test set, so the test score would be too optimistic.
# 4. **Correlated features:** `estimatedTimeWithAI` and `estimatedTimeWithoutAI` have a 0.86 correlation, so using both would make the coefficients hard to interpret.
# 5. **Small and limited data:** only completed tasks have a time value, so the model only represents completed tasks, not incomplete or ditched tasks. Also, `priorExperience` has a weak correlation with the target (-0.09), so most of the prediction may come from the developer's estimate.
# 
# ### How are you planning to address these challenges?
# 
# 1. I will only use features that are available before the task starts and drop the three post-task columns.
# 2. I will train the model on the log of the time (`np.log1p`) and convert the predictions back to minutes (`np.expm1`). I will keep the outliers because long tasks are real tasks, and I will report MAE (less sensitive to outliers) together with RMSE and R².
# 3. I will split the data by developer using `GroupShuffleSplit` (80% train / 20% test) on `developerId`, and use `GroupKFold` cross-validation on the training set, so developers in the test set are never seen during training.
# 4. I will use only one estimate per task, the one that matches the assigned condition, instead of both estimate columns.
# 5. I will clearly state that the results only apply to completed tasks, use cross-validation to get a more stable score from the small dataset, and compare the model with the median baseline.

# ## Machine Learning Model
# 
# In this section, I will build a Linear Regression model on the METR data to predict `initialImplementationTime` from AI usage, prior experience, and the developer's estimated time.

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ### Step 1: Prepare Features and Target

# In[ ]:


ml_df = metr_complete_df.copy()

# TODO: create `estimatedTime` = estimatedTimeWithAI for AI-allowed tasks,
#       estimatedTimeWithoutAI for AI-disallowed tasks (hint: np.where)

# TODO: X = ml_df[["treatmentAssignment", "priorExperience", "estimatedTime"]]
# TODO: y = np.log1p(ml_df["initialImplementationTime"])
# TODO: groups = ml_df["developerId"]


# ### Step 2: Split Train and Test Sets by Developer

# In[ ]:


# TODO: GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# TODO: get train_idx, test_idx from splitter.split(X, y, groups)
# TODO: X_train, X_test, y_train, y_test, groups_train
# TODO: check that no developerId appears in both train and test


# ### Step 3: Baseline Model

# In[ ]:


# TODO: DummyRegressor(strategy="median"), fit on X_train, y_train
# TODO: predict on X_test, convert back to minutes with np.expm1
# TODO: compute MAE, RMSE, R² in minutes


# ### Step 4: Linear Regression Model

# In[ ]:


# TODO: ColumnTransformer: OneHotEncoder(drop="first") for treatmentAssignment,
#       passthrough for priorExperience and estimatedTime
# TODO: Pipeline([("prep", preprocessor), ("model", LinearRegression())])
# TODO: cross_validate on the training set with GroupKFold(n_splits=5), groups=groups_train
# TODO: print mean and std of the CV scores


# ### Step 5: Evaluate on the Test Set

# In[ ]:


# TODO: fit the pipeline on the full training set
# TODO: predict on X_test, convert back to minutes with np.expm1
# TODO: compute MAE, RMSE, R² and compare with the baseline in one table
# TODO: plot predicted vs actual time (log scale)


# ### Step 6: Interpret the Coefficients

# In[ ]:


# TODO: get feature names from the preprocessor and coefficients from the model
# TODO: show them in a DataFrame
# TODO: because y is log(time), exp(coef) - 1 is the % change in time per unit of the feature


# ### Machine Learning Results
# 
# TODO: summarize the model performance compared with the baseline, what the AI-allowed coefficient says about completion time, and the limitations (completed tasks only, small number of developers).

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

# In[144]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

