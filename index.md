---
layout: single
author_profile: true
permalink: /
title: "About Me"
---

I'm a Senior Software Engineer focused on backend systems, AWS infrastructure, data platforms, and agentic AI. My career began in data science and analytics before moving into software engineering, where I now build the systems, infrastructure, and developer tools that make complex workflows reliable and maintainable.

Currently, I'm a Senior Developer at InfoBate. I established the team's AWS CDK codebase and CI/CD pipeline in Python from scratch and have designed production data pipelines using services including Lambda, Step Functions, S3, Glue, Athena, SageMaker, RDS, and CloudWatch.

More recently, I designed and built an internal multi-agent software engineering platform that turns natural-language requests into reviewed GitHub pull requests across multiple production repositories. The system uses LangGraph and the Claude Agent SDK, with human approval gates, scoped tools, automated evaluations, and guardrails around agent behavior. I also built an MCP server in Python that gives agents persistent access to tagged notes, reusable prompts, and project context across sessions.

My work often sits at the intersection of technical architecture and practical business needs. I partner directly with clients and teammates to turn ambiguous requirements into shipped software, from database schemas and optimized queries to backend workflows and React-based interfaces.

Before InfoBate, I was a Software Development Engineer at Amazon, where I worked on AWS data pipelines for Amazon Go processing more than two million events each day. I also designed a resumable Step Functions workflow that eliminated redundant processing and saved approximately 13 hours of compute per execution.

Prior to Amazon, I was a Decision Scientist at Ibotta, where I automated recurring data workflows with Airflow, migrated legacy processes to PySpark, and improved the performance and maintainability of analytics systems.

I earned a Master of Information and Data Science from UC Berkeley, where my graduate work included deep learning, natural-language processing, and an audio super-resolution thesis built with PyTorch. Some of that earlier coursework and experimentation is featured in this portfolio.

I started my path in data science during undergrad, then spent a year teaching English in Vietnam before returning to the US and earning my Master of Information and Data Science at UC Berkeley. My graduate ML coursework is showcased in this portfolio.

Outside of work, I spend as much time as possible rock climbing, skiing, mountaineering, and exploring the mountains. I also converted a Dodge ProMaster into a campervan, one of my favorite examples of the process I enjoy most: learn, plan, build, troubleshoot, and repeat.


---

# Skills & Technologies

**Cloud & Infrastructure**
AWS CDK (Python), Lambda, Step Functions, S3, Glue, Athena, SageMaker, RDS, DynamoDB, CloudWatch, SES, API Gateway · Infrastructure-as-code · CI/CD pipelines

**Languages**
Python · TypeScript · JavaScript · SQL

**Data & ML**
PostgreSQL · PySpark · Airflow · data pipeline design · ML infrastructure

**Frontend**
React · react-admin

---

# Portfolio

The projects below are from my graduate work at UC Berkeley. For information about my more recent professional work, feel free to reach out at [pattidegner@gmail.com](mailto:pattidegner@gmail.com) or connect on [LinkedIn](https://www.linkedin.com/in/patricia-degner/).

## Machine Learning Projects (Python)

### [Mountain Project Sentiment Analysis using Transfer Learning and DistilBERT](/Machine_Learning/mp/mountain_project.md)

An end-to-end deep learning project for sentiment analysis of natural language. The goal is to label sentiment of Mountain Project forum posts to determine what gear climbers consider best. Covers webscraping, data cleaning, model selection, and final analysis. Solo project.

<img src="images/MP.png?raw=true">

---

### [Audio Super-Resolution — Thesis Project](/Machine_Learning/DeciBull/summary.md)

Uses GANs and Autoencoders to upsample low-quality audio in real time. Team project.

<img src="/Machine_Learning/DeciBull/waves.png?raw=true">

---

### [Car Damage Detection Part 1 — Image Classification](/Machine_Learning/car_damage/sup_summary.md)

Uses traditional ML (KNN, Naive Bayes) and CNNs (with and without dropout) to classify images of cars as damaged or whole. Solo project.

### [Car Damage Detection Part 2 — Damage Localization](/Machine_Learning/car_damage/unsupervised_learning/writeup/unsup_summary.md)

Uses Mask R-CNN to identify the location of damage on a car. Team project.

<img src="/Machine_Learning/car_damage/car_damage_cover.png?raw=true">

---
