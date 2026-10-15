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
# This project aims to use metrics such as coding time, AI usage, bugs reported, and task success to examine developer productivity and its relationship with AI development tools. In addition, the project will examine real-world AI use development activity through Copilot-authored pull requests, using metrics such as pull request merge time, commits, and changed files.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# I'd like to ask how AI usage is changing software developement in term of developer productivity, developer experience, and technology trends.
# The analysis will:
# - Analyze how AI usage is related to implementation time, post-review work, perceived effort, and task completion.
# 
# - Examine developers’ AI usage, trust, frustrations, perceived productivity, and experience.
# 
# - Examine real-world AI-assisted development activity through Copilot-authored pull requests, focusing on merge time, commits, and changed files.

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
# - Developers may perceive AI as improving productivity even when measured productivity shows mixed results.
# 
# - Higher AI usage may also introduce additional debugging or quality-related concerns.
# - Copilot-authored pull requests may show different merge-time patterns depending on the size and complexity of the contribution.
# 
# This project aims to study about the correlation and relationships, rather than claiming that AI directly changes developer productivity. 

# ## Data Sources
# 
# This project will use three data sources:
# 
# - METR Developer Productivity Study (CSV) will be used to collect developer productivity which includes tasks completed with and without AI assistance.
# 
# - Stack Overflow Developer Survey 2025 (CSV) will be used to examine developers' use of AI tools, trust in AI output, etc. 
# 
# - GitHub REST API (API) will be used to retrieve real-world development activity related to AI coding agents (GitHub Copilot).
# 
# The datasets will not be directly merged but instead, their results will be used to compare in the final analysis to provide different perspectives on how AI is affecting software development.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# - METR Developer Productivity Study: this data source provides a list of developers with AI and non-AI usage. I can use metrics like `implementation time`, `post-review time`, `task completion`, and `perceived effort` to compare productivity between these two groups. Bar charts may be used to support the analysis through visualization.
# 
# - Stack Overflow Developer Survey 2025: used to examine `AI usage`, `trust`, `frustrations`, `perceived productivity`, and `developer experience`. This can be used to compare responses across different groups of developers and examine how developers perceive the impact of AI tools on their productivity.
# 
# - GitHub REST API: used to examine real-world software development activity involving AI coding agents. I will retrieve Copilot-authored pull request data, including `creation time`, `merge time`, `commits`, and `changed files`. These metrics can be used to identify patterns in AI usage development activity on real-world GitHub repositories, such as how long Copilot-authored pull requests take to be merged and how pull request size or complexity relates to merge time. Bar charts and distribution plots may be used to visualize these patterns.

# # Setup and Import Libraries
# 

# In[26]:


# Import libraries
import numpy as np
import pandas as pd
import requests


# # Import dataset
# 

# ### Import METR Data - Open Source Developer Productivity Data

# In[21]:


metr_df = pd.read_csv("data/raw/osdp.csv")
metr_df.shape


# In[22]:


metr_df.head()


# In[23]:


metr_df.info()


# ### Import Stack Overflow Data
# 
# Because the original dataset is over 100MB, I couldn't push it to GitHub without setting up GitHub LFS, please download the data source from [Stack Overflow Developer Survey 2025](tps://github.com/StackExchange/Survey/blob/main/packages/archive/2025/results.csv) and rename it to `stackoverflowdata.csv` and move it to `data/raw`

# In[20]:


stackoverflow_df = pd.read_csv("data/raw/stackoverflowdata.csv")
stackoverflow_df.shape


# In[24]:


stackoverflow_df.head()


# In[25]:


stackoverflow_df.info()


# ## Prepare Stack Overflow Data
# The Stack Overflow Developer Survey contains a lot of useful information, but it also includes columns that are not relevant to this project. For this part, I will keep only the columns that are useful for my analysis.

# In[79]:


# Filter out the columns that are used in the analysis
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


# ### Import Data From GitHub REST API

# For the GitHub analysis, I will use the GitHub REST API to collect pull requests authored by GitHub Copilot. I will examine their merge time and use the number of commits and changed files as additional indicators of pull request size and complexity.

# In[ ]:


# Import Copilot-authored merged pull requests from GitHub
url = "https://api.github.com/search/issues"

params = {
    "q": "is:pr is:merged author:@copilot",
    "sort": "created",
    "order": "desc",
    "per_page": 20
}

response = requests.get(url, params=params)
data = response.json()

copilot_prs = []

# Loop through each pull request returned by the search
for pr in data["items"]:
    pr_url = pr["pull_request"]["url"]
    pr_response = requests.get(pr_url)

    if pr_response.status_code == 200:
        detail = pr_response.json()
        copilot_prs.append({
            "title": detail["title"],
            "repo_url": pr["repository_url"],
            "created_at": detail["created_at"],
            "merged_at": detail["merged_at"],
            "commits": detail["commits"],
            "changed_files": detail["changed_files"]
        })

copilot_df = pd.DataFrame(copilot_prs)

copilot_df.head()
print(detail)


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# [Open Source Developer Productivity - Late 2025 Follow-on Study Results ](https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs)
# 
# [Stack Overflow Developer Survey](https://github.com/StackExchange/Survey)
# 
# [GitHub REST API](https://docs.github.com/en/rest?apiVersion=2026-03-10)

# ## AI Use Disclosure
# 
# I used AI tools to refine project scope and data analysis ideas, improve written explanations. I also used AI to help compare my project scope with the available datasets to ensure the data I selected could actually help answer the questions. I reviewed the final analysis, interpretations, and conclusions myself.

# In[4]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

