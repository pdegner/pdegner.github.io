---
title: "Measuring The Effect of Difficulty Labels on Problem Solving"
output:
  pdf_document: default
  html_document: default
---
##### Qian (Cathy) Deng, Patti Degner, and Heather Heck

# Motivation

Some people like a challenge; others prefer to be told exactly what to do. Usually, the difficulty of a problem is something we can only truly determine on our own.  However, we are often given some indication of a problem’s difficulty beforehand: by the problem's reputation or source, by the opinion of others who worked on it previously, or (in the case of exams like the GMAT) because we expect the problems to get harder as we proceed.
The question that our experiment will answer is: How does the expectation of the difficulty of a problem change people's ability to solve it?  Does the indication of difficulty breed discouragement if a subject is told the problem is hard? Does it cause carelessness if the subject is told the problem is easy?  Conversely, does being told about difficulty inspire confidence if told the problem is easy, or extra determination if told it is hard?
The answer to this question would be useful for educational institutions, and by those in management. The large test prep industry can use this information to help individuals who wish to improve their performance on standardized tests. Students may be interested in learning if knowing the difficulty of a problem could change their ability to do well on standardized tests.  The findings from this study could be applied in education more broadly: teachers could improve their students' accuracy rate on tests and homework by providing the difficulty. Similarly, many professions require problem solving abilities; companies may want to know if there is a benefit to revealing the true difficulty of a problem. If so, they may want to inform their employees of the difficulties beforehand to improve their employees' ability to solve the problems.  If employees spend more time on difficult problems and increase accuracy by spending less time on easier tasks, the result could be a more efficient workplace.
Prior research notes that adaptive tests, tests that get harder as you go along, tend to improve students’ learning outcomes (Heitmann). On the other hand, additional research suggests that the adaptive tests create anxiety when it comes to difficult questions (Ponsoda). Our experiment can help us understand whether the performance improvements from the difficulty-adaptive tests are due entirely to the adaptiveness of the tests, or if the psychological experience of having some idea of question difficulty can play a helpful role as well. In the case of adaptive tests, the problems are often not given a difficulty label directly, but students are aware that the test will become harder as it goes. We plan to label the questions directly because we want to know if it is knowledge of the problem difficulty that has an effect on performance. 
    The experiment was delivered via online survey. Half of the survey respondents were randomly assigned to control and half were randomly assigned to treatment.  Control consisted of seeing 15 critical thinking problems at once without difficulty listed.  Treatment subjects were shown the same 15 questions at once with the difficulty rating included.  The difficulty levels were as follows: Easy, Medium, Hard, and Very Hard.  The questions are shown below.  The choices offered are provided in parentheses, with the correct answer bolded.


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


```{r, echo = FALSE}
library(tidyverse)
library(sandwich)
library(stargazer)
```

## The Questions    
  1. EASY - How many continents are there in the world? (5,6,*7* or 8)
  
  2. EASY - What is regulation height for a basketball hoop? (*10ft*, 11ft, 12ft, 13ft)
  
  3. EASY - What is the sum of the angles of a triangle? (120 degrees, 160 degrees, *180 degrees*, 200 degrees)
  
  4. MEDIUM - Who wrote the book Frankenstein? (*Mary Shelley*, Maurice Sendak, Edgar Allan Poe, Charles Dickens)
  
  5. MEDIUM - In Harry Potter, what does the Imperius Curse do? (tortures, kills, immobilizes, *controls*) 
  
  6. MEDIUM - How many sides are there in a hexagon? (6, 8, *10*, 12)
  
  7. MEDIUM - Select the correct way to write the asterisked part of the below sentence: Hospitals are increasing the hours of doctors, ***significantly affecting the frequency of surgical errors, which already are a cost to hospitals of*** millions of dollars in malpractice lawsuits. ((A) significantly affecting the frequency of surgical errors, which already are a cost to hospitals of *(B) significantly affecting the frequency of surgical errors, which already cost hospitals* (C) significantly affecting the frequency of surgical errors, already with hospital costs of (D) significant in affecting the frequency of surgical errors, and already costs hospitals)
  
  8. HARD - Who invented the rabies vaccination? (*Louis Pasteur*, Louis Cooper, Jonas Salk, John Robbins)
  
  9. HARD - One day, a person went to a horse racing area. Instead of counting the number of humans and horses, he counted 74 heads and 196 legs. How many humans and horses were there? (37 humans and 98 horses, *24 horses and 50 humans*, 31 horses and 74 humans, 24 humans and 50 horses)
  
  10. HARD - What is the only US state that only touches one other state? (Florida, Michigan, Rhode Island, *Maine*)
  
  11. HARD -Which of these cars did James Bond not drive in any of the James Bond films? (Bentley, Toyota, *Acura*, Mercury)
  
  12. HARD - The average temperature of the last six days is 44 degrees. The median temperature is 36. If Tuesday was the warmest of the six days, what was the lowest possible temperature Tuesday could have had? (84, 64, 48, *60*)
  
  13, VERY HARD - How many ways can the letters of the word PUZZLE be scrambled so that the first and the last letters are both vowels? (12, *24*, 144, 720)
  
  14. VERY HARD - Which gas is formed when a hydrogen bomb is detonated? (Hydrogen, *Helium*, Methane, Uranium Dioxide)
  
  15. VERY HARD - Rephrase this sentence so it has the same meaning: "If you don't keep calm, I shall shoot you," he said to her in a calm voice.  (He warned her to shoot if she didn't keep quiet calmly, *He warned her calmly that he would shoot her if she didn't keep quiet*, He said calmly that I shall shoot you if you don't be quiet, Calmly he warned her that be quiet or else he will have to shoot her)

## Precalculating Power:
If the average respondent answers 10/15 correct without difficulty lables, and we have an effect size of 1 and standard deviation of 2, then we would need 215 respondents to see statistical power. 

```{r, echo=FALSE}
possible.ns <- seq(from=25, to=525, by=10)     # The sample sizes we'll be considering
powers <- rep(NA, length(possible.ns))           # Empty object to collect simulation estimates
alpha <- 0.05                                    # Standard significance level
sims <- 500                                      # Number of simulations to conduct for each N

#### Outer loop to vary the number of subjects ####
for (j in 1:length(possible.ns)){
  N <- possible.ns[j]                              # Pick the jth value for N
  
  significant.experiments <- rep(NA, sims)         # Empty object to count significant experiments
  
  #### Inner loop to conduct experiments "sims" times over for each N ####
  for (i in 1:sims){
    Y0 <- rnorm(n=N, mean=2/3*100, sd=2/15*100)    # control potential outcome
    tau <- 1/15*100                                # Hypothesize treatment effect
    Y1 <- Y0 + tau                                 # treatment potential outcome
    Z.sim <- rbinom(n=N, size=1, prob=.5)          # Do a random assignment
    Y.sim <- Y1*Z.sim + Y0*(1-Z.sim)               # Reveal outcomes according to assignment
    fit.sim <- lm(Y.sim ~ Z.sim)                   # Do analysis (Simple regression)
    p.value <- summary(fit.sim)$coefficients[2,4]  # Extract p-values
    significant.experiments[i] <- (p.value <= alpha) # Determine significance according to p <= 0.05
  }
  
  powers[j] <- mean(significant.experiments)       # store average success rate (power) for each N
}
plot(possible.ns, powers, ylim=c(0,1), xlab = "Sample Size", ylab = "Power",
    main = "Sample Size Needed")
abline(v = 210)
abline(h = 0.95)
```


  If we estimated the impact (tau) of the difficulty ratings as 2 questions or 2/15, we would need approximately 65 respondents.

```{r, echo=FALSE}
possible.ns <- seq(from=25, to=525, by=10)     # The sample sizes we'll be considering
powers <- rep(NA, length(possible.ns))           # Empty object to collect simulation estimates
alpha <- 0.05                                    # Standard significance level
sims <- 500                                      # Number of simulations to conduct for each N

#### Outer loop to vary the number of subjects ####
for (j in 1:length(possible.ns)){
  N <- possible.ns[j]                              # Pick the jth value for N
  
  significant.experiments <- rep(NA, sims)         # Empty object to count significant experiments
  
  #### Inner loop to conduct experiments "sims" times over for each N ####
  for (i in 1:sims){
    Y0 <- rnorm(n=N, mean=2/3*100, sd=2/15*100)    # control potential outcome
    tau <- 2/15*100                                # Hypothesize treatment effect
    Y1 <- Y0 + tau                                 # treatment potential outcome
    Z.sim <- rbinom(n=N, size=1, prob=.5)          # Do a random assignment
    Y.sim <- Y1*Z.sim + Y0*(1-Z.sim)               # Reveal outcomes according to assignment
    fit.sim <- lm(Y.sim ~ Z.sim)                   # Do analysis (Simple regression)
    p.value <- summary(fit.sim)$coefficients[2,4]  # Extract p-values
    significant.experiments[i] <- (p.value <= alpha) # Determine significance according to p <= 0.05
  }
  
  powers[j] <- mean(significant.experiments)       # store average success rate (power) for each N
}

plot(possible.ns, powers, ylim=c(0,1), xlab = "Sample Size", ylab = "Power",
    main = "Sample Size Needed")
abline(v = 65)
abline(h = 0.95)
```

## EDA

Our responses were heavily clusterd by region.


```{r, echo=FALSE}
df <- read_csv("results_clean.csv")
df <- df[,!(names(df) %in% c("RecordedDate", "ResponseId", "Total", "Easy", "Medium", "Hard", "Very_Hard"))]
df <- df %>% rename(duration = `Duration (in seconds)`)
df$Finished <- as.numeric(df$Finished)
df$group <- as.numeric(df$group=="treatment")
```


```{r, echo=FALSE}
counts = c(sum(df$mid_atlantic, na.rm = TRUE),sum(df$midwest, na.rm = TRUE),sum(df$mountain, na.rm = TRUE),
           sum(df$new_england, na.rm = TRUE),sum(df$ne_midwest, na.rm = TRUE),sum(df$not_usa, na.rm = TRUE),
           sum(df$pacific, na.rm = TRUE),sum(df$south_atlantic, na.rm = TRUE),sum(df$west_south, na.rm = TRUE))

naming = c("mid_atlantic","midwest","mountain","new_england","ne_midwest",
           "not_usa","pacific","south_atlantic","west_south")

#increase margins
#bottom, left, top, right 
par(mar=c(7,4,4,4))

barplot(counts, names = naming,
  ylab = "Number of Responses",
  main = "Number of Responses by Region",
        las = 2, ylim=c(0,70)
       )
```

  Our subjects are on average more educated, younger, and regionally clustered than the general American population. The graphs below show our population distribution compared to the USA distributions. Therefore, the scope of our conclusions should be limited to only those who are similar to our test subjects.  

```{r, echo=FALSE}
# American Education
labels <- c("8th_less", "less_high_school", "hs_grad", "some_college", "associate", "bachelor", "master", "professional", "doctorate")
data <- c(8603,	13372,	62259,	34690,	22738,	49937,	22214,	3136,	4529)
barplot(data,
main = "Level of Education for Americans (in thousands)",
xlab = "Education Level",
ylab = "Count",
col = "darkred", space = 1)
text(seq(1.5,17.5,by=2), par("usr")[3]-0.25, 
     srt = 60, adj= 1, xpd = TRUE,
     labels = labels, cex=0.65)
print(" ")

# Survey Education
labels <- c("Less than high school", "2 year degree", "4 year degree", 'Master\'s degree', "Doctorate", "NA")
data <- df %>%
  group_by(education) %>%
  summarize(no_rows = length(education))
barplot(data$no_rows,
main = "Level of Education for Study Participants",
xlab = "Education Level",
ylab = "Count",
col = "darkblue", space = 1)
text(seq(1.5,12.5,by=2), par("usr")[3]-0.25, 
     srt = 60, adj= 1, xpd = TRUE,
     labels = labels, cex=0.65)
print(" ")

# American Age
labels <- c('Under 18','18 - 24','25 - 34','35 - 44','45 - 54','55 - 64','65 - 74')
data <- c(81892, 21434, 44855, 40660, 41537, 41700, 30366)
barplot(data,
main = "Age of Americans (in thousands)",
xlab = "Age Bracket",
ylab = "Count",
col = "darkred", space = 1)
text(seq(1.5,14.5,by=2), par("usr")[3]-0.25, 
     srt = 60, adj= 1, xpd = TRUE,
     labels = labels, cex=0.65)
print(" ")

# Survey Age
labels <- c('Under 18','18 - 24','25 - 34','35 - 44','45 - 54','55 - 64','65 - 74')
data <- df %>%
  group_by(age) %>%
  summarize(no_rows = length(age))
barplot(data$no_rows,
main = "Age of Study Participants",
xlab = "Age Bracket",
ylab = "Count",
col = "darkblue", space = 1)
text(seq(1.5,14.5,by=2), par("usr")[3]-0.25, 
     srt = 60, adj= 1, xpd = TRUE,
     labels = labels, cex=0.65)
```

## How Accurate were our difficulty labels?

```{r, echo=FALSE}
boxplot(df$Total_Pct,df$Easy_Pct,df$Medium_Pct, df$Hard_Pct, df$Very_Hard_Pct,
        names=c("Total","Easy","Medium","Hard","VeryHard"),
        main = "Score by Difficulty", ylab = "Score (in percentage)", xlab = "Difficulty")
```

## Models

### Simple regressions
```{r, echo=FALSE}
model1 = lm (Total_Pct ~ group, data = df)
cov1 <- vcovHC(model1)
se1 <- sqrt(diag(cov1))

stargazer(model1, se = list(se1), type = 'text', title = "Naive Model")
```

```{r, echo=FALSE}
model2 = lm (Easy_Pct ~ group, data = df)
cov2 <- vcovHC(model2)
se2 <- sqrt(diag(cov2))

model3 = lm (Medium_Pct ~ group, data = df)
cov3 <- vcovHC(model3)
se3 <- sqrt(diag(cov3))

model4 = lm (Hard_Pct ~ group, data = df)
cov4 <- vcovHC(model4)
se4 <- sqrt(diag(cov4))

model5 = lm (Very_Hard_Pct ~ group, data = df)
cov5 <- vcovHC(model5)
se5 <- sqrt(diag(cov5))

stargazer(model1, model2, model3, model4, model5, 
          se = list(se1,se2,se3,se4,se5), 
          type = 'text', title = "Naive Models")
```

  From the naive models, we observe that the treatment effect is significant at the p<0.05 level for both the total score and for the score on hard questions.
  
```{r}
names(df)
head(df)
```

  
### More Models

Note: our dataset was cleaned and tidied further using Excel. The results from that cleaning are used to create the models. 

```{r, echo=FALSE} 
df <- read_csv("excel_clean.csv")


model1 = lm (Total_Pct ~ group, data = df)
cov1 <- vcovHC(model1)
se1 <- sqrt(diag(cov1))

model2n = lm (Total_Pct ~ group + Female + group * Female + age_25_34 + group * age_25_34, data = df)
cov2n <- vcovHC(model2n)
se2n <- sqrt(diag(cov2n))

model3n = lm (Total_Pct ~ group + Female + group * Female + age_25_34 + group * age_25_34 + mid_atlantic + new_england + employeed_FT + income_200 + stress_little_more + stress_alot_more + master, data = df)
cov3n <- vcovHC(model3n)
se3n <- sqrt(diag(cov3n))

stargazer(model1, model2n, model3n,
          se = list(se1, se2n, se3n), 
          type = 'text', title = "All Models")
```

### f-tests

```{r, echo=FALSE} 
anova(model1, model2n, test = 'F')
```


```{r, echo=FALSE} 
anova(model2n, model3n, test = 'F')
```

### LASSO Regression

```{r, echo=FALSE}
df <- read_csv("results_clean.csv")
df <- df[,!(names(df) %in% c("RecordedDate", "ResponseId", "Total", "Easy", "Medium", "Hard", "Very_Hard"))]
df <- df %>% rename(duration = `Duration (in seconds)`)
df$Finished <- as.numeric(df$Finished)
df$group <- as.numeric(df$group=="treatment")
head(df)
```

Below, you can see how the coefficients shrink to 0 with LASSO Regression.

```{r, echo=FALSE}
library(glmnet)
x_vars <- model.matrix(Total_Pct ~  duration + challenge + education + gender + stress + age + group + retired + student + unemployed, df_noloc)[,-1]
y_var <- df_noloc$Total_Pct
fit <- glmnet(x_vars,y_var)
plot(fit)
```

Below is the number of nonzero coefficients, the percent, deviance explained (%dev) and the value of λ
```{r, echo=FALSE}
lambda_seq <- 10^seq(2, -2, by = -.1)
cv_output <- cv.glmnet(x_vars, y_var, alpha = 1, lambda = lambda_seq)
cv_output$lambda.1se
plot(cv_output) # number of nonzero coefficients (Df), the percent (of null) deviance explained (%dev) and the value of λ
```

Now, compare the model where lambda minimizes the mean squared error, vs. the model where the mean squared error is within one standard deviation of the minimum, and the model is as small as possible. 

```{r, echo=FALSE}
coef(cv_output, s="lambda.min")
print("--------------------------------------------")
coef(cv_output, s="lambda.1se")
```

Now, use lambda where the mean squared error is within one standard deviation of the minimum and calculate accuracy of the model by creating a validation set. Below are the coefficients of the model. Note that none were removed completely.

```{r, echo=FALSE}
# https://www.rstatisticsblog.com/data-science-in-action/lasso-regression/
# Make a train and test set to find best model 

set.seed(7)
# Get options for lambda
lambda_seq <- 10^seq(2, -2, by = -.1)

# Create train and test data
train = sample(1:nrow(x_vars), nrow(x_vars)/2)
x_test = (-train)
y_test = y_var[x_test]

# Find best lambda based on train data
cv_output <- cv.glmnet(x_vars[train,], y_var[train], alpha = 1, lambda = lambda_seq)
best_lam <- cv_output$lambda.min #or cv_output$lambda.1se

# Rebuild model with best lambda
lasso_best <- glmnet(x_vars[train,], y_var[train], alpha = 1, lambda = best_lam)

# Inspect beta coefficinets
coef(lasso_best)
```

The RSQ for the LASSO model is just 16.2%. Therefore, it would be better to use OLS regression in this case. All conclusions will be drawn from the OLS regresion models. 
```{r, echo=FALSE}
# How did the model do?
pred <- predict(lasso_best, s = best_lam, newx = x_vars[x_test-1,])
final <- cbind(y_var[x_test-1], pred)

actual <- final[,1]
predicted <- final[,2]
rss <- sum((predicted-actual)^2)
tss <- sum((actual - mean(actual))^2)
rsq <- 1-rss/tss
rsq
```

# Conclusion

In summary, our experiment showed a statistically and practically significant effect of showing difficulty labels to test takers. Additionally, it showed a significant effect of informing test takers of the difficulty of “hard” questions in particular.
These conclusions can help explain behaviors in certain real world situations. For instance, it is possible that some high scorers on the GMAT attain their success by evaluating the difficulty of a question prior to attempting to solve it. This expectation changes their approach to the question (e.g. the amount of “second-guessing”) and ultimately improves accuracy. 
In addition, we can apply these conclusions to improve performance in various contexts. A manager in a business can motivate employees to perform a task at a higher standard by indicating that it is a hard problem. Indeed, the significance of the treatment effect on the total score in the survey suggests that comprehensive conversations about difficulty can be useful across the full range of difficulty, again because it could help individuals assess the relative priority of tasks, taking into account capacity demands and level of complexity. Based on the results, we might also hypothesize that knowing “hard” questions are hard can be motivating in itself. However, this effect seems to diminish at the highest level of difficulty, “very hard” - practically speaking, the motivation can push people to strive beyond a baseline level, but not exceed the true limits of their ability and time.
Future enhancements to our experiment could include some of the following options.  We would be interested in making questions open-ended.  Multiple choice questions allow survey respondents to use the process of elimination.  They may eliminate answer choices that seem ‘too easy’ if a question was labeled hard.  We could eliminate some of that risk by switchen to open-ended questions.  If given more time, we could also better calibrate the difficulty of our questions.  If we ran a robust pilot, we could separate questions that were only answered correctly by those who did well on the survey, and use those questions as the hard questions.  Alternatively, we could eliminate questions that almost everyone answered correctly.  If we had more resources and time, we would have gathered a larger sample of respondents and a more diverse sample of respondents.  We could add a covariate that measured survey takers' background.  This covariate could focus on their area of work, their major in college, or some set of questions that could isolate what they were a subject matter expert in, to account for variations caused by people’s greater ability to answer questions that they have expertise in.
It would be interesting to break this experiment into many sub-experiments.  We could have one experiment that was only math questions, one experiment that was only grammar questions, etc., to see if the effect is more significant in some fields than others.  We could also go the other route and provide a survey with questions that tested more diverse subject matters: the more diverse the questions, the more likely it is that respondents would have to reach beyond their existing domain expertise.
One final possible enhancement would be a longer survey with more questions, which would allow us to measure the impact of difficulty labels on a more granular scale.  With the 15 questions we used, each additional question answered correctly added 6.7% to the total percentage score.  If we had 30 questions, it would cut that increase in half.


### Bibliography:
Heitmann, Svenja, et al. “Testing Is More Desirable When It Is Adaptive and Still Desirable When Compared to Note-Taking.” Frontiers in Psychology, Frontiers Media S.A., 18 Dec. 2018, www.ncbi.nlm.nih.gov/pmc/articles/PMC6305602/.

Ponsoda, Vicente, et al. “The Effects of Test Difficulty Manipulation in Computerized Adaptive Testing and Self-Adapted Testing.” Applied Measurement in Education, vol. 12, no. 2, 1999, pp. 167–184., doi:10.1207/s15324818ame1202_4.

US Census Bureau. “Age and Sex Composition in the United States: 2018.” The United States Census Bureau, 11 July 2019, www.census.gov/content/census/en/data/tables/2018/demo/age-and-sex/2018-age-sex-composition.html.

US Census Bureau. “Educational Attainment in the United States: 2019.” The United States Census Bureau, 30 Mar. 2020, www.census.gov/data/tables/2019/demo/educational-attainment/cps-detailed-tables.html.



