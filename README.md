# Final-Project-Template
<!-- Edit the title above with your project title -->

## Project Overview

## Self Assessment and Reflection

<!-- Edit the following section with your self assessment and reflection -->

### Self Assessment
<!-- Replace the (...) with your score -->

| Category          | Score    |
| ----------------- | -------- |
| **Setup**         | 10 / 10 |
| **Execution**     | 20 / 20 |
| **Documentation** | 10 / 10 |
| **Presentation**  | 30 / 30 |
| **Total**         | 70 / 70 |

### Reflection

#### What went well?
I was able to combine three different types of data sources: a research dataset METR, a survey of Stack Overflow 2025, and a Artificial Analysis dataset. 
Each one answered a different part of my question. 

The results from different datasets supported each other. I handled data quality. For instance, I checked missing/duplicate values. In the machine learning part, I split train and test sets by developer with `GroupShuffleSplit`, so the model was always tested on developers it had never seen.

#### What did not go well?
The Stack Overflow csv file is over 100MB, so I couldn't push it to GitHub. Therefore, I noted in my notebook that anyone running the notebook has to download it manually.

I also needed to reconsider and change my data sources and project scope several times early on, and I had to rework the data preparation step to meet the rubric, which took time away from the analysis.

In my opinion, the machine learning model didn't perform well. The test MAE was about 70 minutes, which is large compared to the median test task (45 minutes).

#### What did you learn?
From this project, I need to carefully consider a lower error on the test set because lower error doesn't always mean a better model. I had to compare the task times in each set to see that the test tasks were just shorter.

I also learned how to complete an end-to-end project, where I had to define the project scope, find datasets, and decide which questions needed to be answered. I learned to clean data, perform EDA, visualize data, answer business questions, and build machine learning models (linear regression and random forest) on the dataset.

#### What would you do differently next time?
If I were to do differently, I would lock in my data sources and project scope earlier. 

I may consider using cross-validation with instead of a single train/test split, so the results wouldn't depend on which developers end up in the test set.

---

## Getting Started
### Installing Dependencies

To ensure that you have all the dependencies installed, and that we can have a reproducible environment, we will be using `pipenv` to manage our dependencies. `pipenv` is a tool that allows us to create a virtual environment for our project, and install all the dependencies we need for our project. This ensures that we can have a reproducible environment, and that we can all run the same code.

```bash
pipenv install
```

This sets up a virtual environment for our project, and installs the following dependencies:

- `ipykernel`
- `jupyter`
- `notebook`
- `black`
  Throughout your analysis and development, you will need to install additional packages. You can can install any package you need using `pipenv install <package-name>`. For example, if you need to install `numpy`, you can do so by running:

```bash
pipenv install numpy
```

This will update update the `Pipfile` and `Pipfile.lock` files, and install the package in your virtual environment.

## Helpful Resources:
* [Markdown Syntax Cheatsheet](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
* [Dataset options](https://it4063c.github.io/guides/datasets)