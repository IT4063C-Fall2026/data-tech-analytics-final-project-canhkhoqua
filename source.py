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
# This project aims to utilize data metrics such as coding hours, AI usage, bugs reported, task success, etc to learn and examine developers productivity. Besides, this project will also examine which technologies topics are receiving increased developer attention since the widespread adoption of generative AI.

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
# - Identify project areas receiving increased developer activity.

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
# 
# This project aims to study about the correlation and relationships, rather than claiming that AI directly changes developer productivity. 

# ## Data Sources
# 
# This project will use three data sources:
# 
# - METR Developer Productivity Study
# 
# - Stack Overflow Developer Survey 2025
# 
# - GitHub Innovation Graph
# 
# The datasets will not be directly merged but instead, their results will be used to compare in the final analysis to provide different perspectives on how AI is affecting software development.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# - METR Developer Productivity Study: this data source provides a list of developers with AI and non-AI usage, I can use metrics like `implementation time`, `post-review time`, `task completion`, and `perceived effort` to compare productivity between these two groups. Bar chart may be used to support the analysis via visualization.
# 
# - Stack Overflow Developer Survey 2025: used to examine `AI usage`, `trust`, `frustrations`, `perceived productivity`, and `developer experience`. This can be used to compare responses across  
# 
# - GitHub Innovation Graph: used to analyze repository topics and then identify which technologies and topics are receiving increased developer activity before and after the widespread adoption of AI. Trend charts will be used to show changes by quarter or year.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# [Open Source Developer Productivity - Late 2025 Follow-on Study Results ](https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs)
# 
# [Stack Overflow Developer Survey](https://github.com/StackExchange/Survey)
# 
# [GitHub Innovation Graph](https://github.com/github/innovationgraph/tree/main)
# 

# ## AI Use Disclosure
# 
# I used AI tools to refine project scope and data analysis ideas, improve written explanations. I also used AI to help compare my project scope with the available datasets to ensure the data I selected could actually help answer the questions. I reviewed the final analysis, interpretations, and conclusions myself.

# In[4]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

