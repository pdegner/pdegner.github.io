## Introduction

The goal of this study is to understand determinants of crime in North Carolina given a dataset from 1987, and recommend a policy to help reduce the crime rate. Our research questions is: "Which factors increase the crime rate?" We need to determine which covariates affect the crime rate (crmrte) the most, making it our dependent variable. 


```R
#Install required packages
install.packages("car")
install.packages("lmtest")
install.packages("sandwich")
install.packages("imputeTS")
install.packages("stargazer")
install.packages("effsize")
install.packages("sandwich")
install.packages("QuantPsyc")
```

    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpK8jknK/downloaded_packages
    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpK8jknK/downloaded_packages
    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpK8jknK/downloaded_packages
    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpK8jknK/downloaded_packages
    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpK8jknK/downloaded_packages
    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpK8jknK/downloaded_packages
    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpK8jknK/downloaded_packages
    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpK8jknK/downloaded_packages



```R
#Import libraries
library(car)
library(lmtest)
library(sandwich)
library(effsize)
library(imputeTS)
library(stargazer)
library(sandwich)
library(QuantPsyc)
```

    Loading required package: carData
    Loading required package: zoo
    
    Attaching package: ‘zoo’
    
    The following objects are masked from ‘package:base’:
    
        as.Date, as.Date.numeric
    
    Registered S3 method overwritten by 'xts':
      method     from
      as.zoo.xts zoo 
    Registered S3 method overwritten by 'quantmod':
      method            from
      as.zoo.data.frame zoo 
    Registered S3 methods overwritten by 'forecast':
      method             from    
      fitted.fracdiff    fracdiff
      residuals.fracdiff fracdiff
    
    Attaching package: ‘imputeTS’
    
    The following object is masked from ‘package:zoo’:
    
        na.locf
    
    
    Please cite as: 
    
     Hlavac, Marek (2018). stargazer: Well-Formatted Regression and Summary Statistics Tables.
     R package version 5.2.2. https://CRAN.R-project.org/package=stargazer 
    
    Loading required package: boot
    
    Attaching package: ‘boot’
    
    The following object is masked from ‘package:car’:
    
        logit
    
    Loading required package: MASS
    
    Attaching package: ‘QuantPsyc’
    
    The following object is masked from ‘package:base’:
    
        norm
    


First, we will load the data set and do some initial cleaning. 


```R
#Import dataframe
df <- read.csv("crime_v2.csv")
df <- na.omit(df) #removes empty rows at the end. 
```

## Data Cleaning
First, we convert probability of conviction (prbconv) to numeric, remove the year since it is constant, remove a duplicate row, set county to be the index and remove it.


```R
df$prbconv = as.numeric(as.character(df$prbconv))
crime = subset(df, select = -year)
crime = crime[-88,]
rownames(crime) <- crime$county
crime$county <- NULL
```

An initial analysis of the histograms and distribution of the data helps us to determine which variables need to be transformed in any manner. We are not going to show this in the report because it is a lot of graphs, but uncommenting and running this code will show them. 


```R
#create histograms
# for (col in 1:ncol(crime)) {
#   hist(crime[,col], main = names(crime)[col])
# }

#create scatterplots
# for (col in 1:ncol(crime)) {
#   plot(crime$crmrte~ crime[,col], main = names(crime)[col])
# }

```

These histograms and scatterplots revealed some unusual data points and skewed distributions. We will do some clean up of the data and then re-check the histograms. First, wage for service industry (wser) has a point that is 10 times greater than most. We divide this point by 10 because we think the decimal was put in the wrong place.


```R
row_10x_wser <- which(df$wser>2000)
crime$wser[row_10x_wser] <- crime$wser[row_10x_wser]/10
hist(crime$wser, main= "Histogram of Weekly Wage for the Service Industry", xlab="weekly wage ($)", breaks = 20)
plot(crime$crmrte ~ crime$wser, main = "Plot of Weekly wage for the Service Industry vs Crime Rate", 
     xlab="weekly wage ($)", ylab="crime rate")
```


![png](output_10_0.png)



![png](output_10_1.png)


After correcting this value we can see that the historgram is quite symmetrical. The scatter plot indicates a weak, positive relationship between the service wage and crime, which is somewhat surprising.

A particular county seems to have a probability of arrest greater than 1, which is impossible. Since we cannot determine the true probability of arrest and the rest of the row provides valuable data, we are going to impute the mean for this variable in its place. This will reduce the variance, but the effect should be slight enough to be negligible while also retaining the other data points. 


```R
invalid_prbarr <- which(crime$prbarr > 1)
mean_prbarr <- mean(crime[-invalid_prbarr,]$prbarr)
crime$prbarr[invalid_prbarr] = mean_prbarr
```

Similarly, 10 rows have probability of conviction greater than 1, which is not a valid probability. We have replaced these values with the means as well.


```R
summary(crime$prbconv)

rem_list <- list(which(crime$prbconv >= 1)) # it is rows 2 10 44 51 56 61 67 84 89 90 

for (col in rem_list) {
  crime$prbconv[col] = NA
}

mean_prbconv = mean(crime$prbconv, na.rm=TRUE)
crime$prbconv <- ifelse(is.na(crime$prbconv), mean_prbconv, crime$prbconv)
```


       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    0.06838 0.34422 0.45170 0.55086 0.58513 2.12121 


The police per capita variable appears to have an outlier.


```R
plot(crime$crmrte ~ crime$polpc, main = "Police per capita vs. Crime", xlab="police per capita", ylab="crime rate")
```


![png](output_16_0.png)



```R
which(crime$polpc > .009) 
```


51


This point also had a probability of arrest greater than 1, a percent minority greater than 1, and an average sentence that was much higher than other points. We think data point 51 has been tampered with and we are throwing it out.


```R
crime = crime[-51,]
```

Percent young male and tax revenue per capita have outliers. At this time, we cannot justify doing anything with the taxpc or pctmle based on our EDA in the graphs below. But, they may have high leverage and need to be removed later. We should look at models with and without these points to make a final call. 


```R
summary(crime$taxpc)
scm_labels <- c("crime rate","tax per capita","probability of conviction","percent young male")
scatterplotMatrix(~crmrte + taxpc + prbconv + pctymle, data = crime, main="Original data", var.labels=scm_labels)
scatterplotMatrix(~crmrte + taxpc + prbconv + pctymle, data = crime[-25,], main="Extreme tax per capita removed", var.labels=scm_labels)
scatterplotMatrix(~crmrte + taxpc + prbconv + pctymle, data = crime[-58,], main="Extreme pct. young male removed", var.labels=scm_labels)
scatterplotMatrix(~crmrte + taxpc + prbconv + pctymle, data = crime[-c(25,58),], main="Extreme values of both removed", var.labels=scm_labels)
```


       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      25.69   30.85   34.96   38.27   41.07  119.76 



![png](output_21_1.png)



![png](output_21_2.png)



![png](output_21_3.png)



![png](output_21_4.png)


From these graphs, we see that there are potentially problematic points but we cannot justify removing the data at this time.

We examine the cleaned plots to determine if anything changed and what further actions need to be taken. 


```R
sum(is.na(crime)) #No missing values
# for (col in 1:ncol(crime)) {
#   hist(crime[,col], main = names(crime)[col])
# }

# for (col in 1:ncol(crime)) {
#   plot(crime$crmrte~ crime[,col], main = names(crime)[col])
# }
```


0


The following variables appear to have skewed data based on our EDA, and would therefore benefit from taking the log of the values: Crime rate (crmrte), Police per capital (polpc), Population density (density), Tax per capita (taxpc), Crime mix (mix), and Percent young male (pctymle). We create a new dataset with these variables transformed and relabled them with a log indicator.


```R
logcrime = crime
logcrime$crmrte = log(crime$crmrte)
logcrime$polpc = log(logcrime$polpc)
logcrime$density = log(logcrime$density)
logcrime$taxpc = log(logcrime$taxpc)
logcrime$mix = log(logcrime$mix)
logcrime$pctymle = log(logcrime$pctymle)

colnames(logcrime)[1] = "log_crmrte"
colnames(logcrime)[6] = "log_polpc"
colnames(logcrime)[7] = "log_density"
colnames(logcrime)[8] = "log_taxpc"
colnames(logcrime)[22] = "log_mix"
colnames(logcrime)[23] = "log_pctymle"
```

For each variable, we once again examined the histogram of the log of the data and the scatterplot with crime rate to ensure the transformation worked as expected.


```R
# for (col in 1:ncol(logcrime)) {
#   hist(logcrime[,col], main = names(logcrime)[col])
# }

# for (col in 1:ncol(logcrime)) {
#   plot(logcrime$log_crmrte~ logcrime[,col], main = names(logcrime)[col])
# }
```

In the scatterplot matrix below, we compared the log data to the regular data. The log data appears to be more linear for each item when compared to crime rate, and less heteroskedastic. After taking the log of "mix" and "crmrte", there appears to be little correlation there. 


```R
scm_labels <- c("crime rate","police per capita","density","tax per capita", "crime mix", "percent young male")
scatterplotMatrix(~crmrte +polpc + density + taxpc +mix +pctymle, data = crime, main="Relationships before taking log", var.labels=scm_labels)
scatterplotMatrix(~log_crmrte +log_polpc + log_density + log_taxpc +log_mix +log_pctymle, data = logcrime, main="Relationships after taking log", var.labels=scm_labels)
```


![png](output_29_0.png)



![png](output_29_1.png)


We know that there may be some imperfect colinearity in the data set. For example, the wage variables may be correlated because if cost of living is high in one county, people in all sectors are probably paid more. We examine the scatterplot matrix of the wage variables to see if there appears to be colinearity.


```R
scatterplotMatrix(~wcon + wtuc + wtrd + wfir + wser + wmfg + wfed + wsta + wloc, 
                  data = logcrime, main = "Relationships among wage variables", 
                  var.labels=c("Const.", "Tran/Util/Con", "Wholesale",
                               "Fin/Real/Ins","Service",
                               "Manuf.","Federal","State Gov.","Local Gov."))
```


![png](output_31_0.png)


There does not appear to be any exact colinearity, but there does appear to be correlation. We examine the correlation matrix to see how large this is.


```R
cor(logcrime[,c(13,14,15,16,17,18,19,20,21)])
```


<table>
<caption>A matrix: 9 × 9 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>wcon</th><th scope=col>wtuc</th><th scope=col>wtrd</th><th scope=col>wfir</th><th scope=col>wser</th><th scope=col>wmfg</th><th scope=col>wfed</th><th scope=col>wsta</th><th scope=col>wloc</th></tr>
</thead>
<tbody>
	<tr><th scope=row>wcon</th><td> 1.00000000</td><td> 0.4432933</td><td>0.577473384</td><td>0.5051639</td><td>0.5569705</td><td>0.38177286</td><td>0.5150461</td><td>-0.027064255</td><td>0.6012692</td></tr>
	<tr><th scope=row>wtuc</th><td> 0.44329328</td><td> 1.0000000</td><td>0.351988290</td><td>0.3253013</td><td>0.4287211</td><td>0.45960866</td><td>0.4020310</td><td>-0.149317766</td><td>0.3123344</td></tr>
	<tr><th scope=row>wtrd</th><td> 0.57747338</td><td> 0.3519883</td><td>1.000000000</td><td>0.6680122</td><td>0.5463144</td><td>0.37220597</td><td>0.6406744</td><td> 0.008159654</td><td>0.5997546</td></tr>
	<tr><th scope=row>wfir</th><td> 0.50516386</td><td> 0.3253013</td><td>0.668012227</td><td>1.0000000</td><td>0.5962259</td><td>0.49701629</td><td>0.6237379</td><td> 0.242884981</td><td>0.5659871</td></tr>
	<tr><th scope=row>wser</th><td> 0.55697049</td><td> 0.4287211</td><td>0.546314431</td><td>0.5962259</td><td>1.0000000</td><td>0.55124829</td><td>0.6091940</td><td> 0.065939399</td><td>0.6091347</td></tr>
	<tr><th scope=row>wmfg</th><td> 0.38177286</td><td> 0.4596087</td><td>0.372205972</td><td>0.4970163</td><td>0.5512483</td><td>1.00000000</td><td>0.5231767</td><td> 0.058733721</td><td>0.4336387</td></tr>
	<tr><th scope=row>wfed</th><td> 0.51504607</td><td> 0.4020310</td><td>0.640674380</td><td>0.6237379</td><td>0.6091940</td><td>0.52317673</td><td>1.0000000</td><td> 0.188390863</td><td>0.5412372</td></tr>
	<tr><th scope=row>wsta</th><td>-0.02706426</td><td>-0.1493178</td><td>0.008159654</td><td>0.2428850</td><td>0.0659394</td><td>0.05873372</td><td>0.1883909</td><td> 1.000000000</td><td>0.1841068</td></tr>
	<tr><th scope=row>wloc</th><td> 0.60126916</td><td> 0.3123344</td><td>0.599754640</td><td>0.5659871</td><td>0.6091347</td><td>0.43363869</td><td>0.5412372</td><td> 0.184106773</td><td>1.0000000</td></tr>
</tbody>
</table>



Some of the pairs of wage variables have fairly high correlation, so we will not favor models that have all or a large number of these variables included. Just a few will suffice to account for the overall wage level in the county.

## Variable selection by Backwards Selection

To determine which variables are correlated with crime, we are going to do backwards selection of our model. The idea is to build a model with all the variables. Then, we will remove the variable that has the largest p-value, re-run the model and compare it to the previous model using the F-test, remove the variable that has the largest p-value, and so on until removing another variable weakens our F-statistic. 

Fortunately, the step() function in R will do this backwards selection for us, using the F-test to compare the newest model with the previous one like so:
1.  We will start with a base model with all variables.
2.  Run F-tests comparing all different models with one variable removed from the base model.
3.  Refit the model without the variable with the largest p-value, so long as the p-value for that variable is larger than .05 and the the F-statistic increases.
4.  Repeat steps 2-3 until deleting a variable will delete a significant variable and weaken our F-statistic.

A drawback to this method is that it relies only data and not on scientific or any other established knowledge in the field. However, we feel this is appropriate in this case because none of us are experts in criminology. 


```R
m1 <- lm(log_crmrte ~ ., data = logcrime)
```


```R
step(m1, direction="backward", test="F")
```

    Start:  AIC=-202.43
    log_crmrte ~ prbarr + prbconv + prbpris + avgsen + log_polpc + 
        log_density + log_taxpc + west + central + urban + pctmin80 + 
        wcon + wtuc + wtrd + wfir + wser + wmfg + wfed + wsta + wloc + 
        log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - log_taxpc    1   0.00545 5.4644 -204.34  0.0658  0.798281    
    - prbconv      1   0.00620 5.4651 -204.33  0.0750  0.785077    
    - wmfg         1   0.00973 5.4686 -204.28  0.1176  0.732758    
    - wcon         1   0.01528 5.4742 -204.19  0.1848  0.668709    
    - wsta         1   0.02644 5.4853 -204.00  0.3196  0.573740    
    - wloc         1   0.05998 5.5189 -203.46  0.7251  0.397548    
    - wtuc         1   0.11349 5.5724 -202.60  1.3722  0.245653    
    - pctmin80     1   0.11992 5.5788 -202.50  1.4499  0.232843    
    <none>                     5.4589 -202.43                      
    - urban        1   0.17291 5.6318 -201.66  2.0906  0.152942    
    - west         1   0.21393 5.6728 -201.01  2.5865  0.112552    
    - wser         1   0.23713 5.6960 -200.65  2.8670  0.095129 .  
    - log_pctymle  1   0.24813 5.7070 -200.48  3.0000  0.087938 .  
    - prbpris      1   0.25244 5.7113 -200.41  3.0521  0.085283 .  
    - wfed         1   0.26847 5.7274 -200.16  3.2459  0.076168 .  
    - wtrd         1   0.27737 5.7363 -200.02  3.3536  0.071574 .  
    - central      1   0.29753 5.7564 -199.71  3.5972  0.062254 .  
    - log_mix      1   0.30065 5.7596 -199.66  3.6349  0.060933 .  
    - wfir         1   0.32132 5.7802 -199.34  3.8848  0.052921 .  
    - avgsen       1   0.56380 6.0227 -195.69  6.8165  0.011168 *  
    - prbarr       1   0.77117 6.2301 -192.67  9.3237  0.003260 ** 
    - log_density  1   1.06104 6.5199 -188.63 12.8284  0.000647 ***
    - log_polpc    1   1.69645 7.1554 -180.35 20.5106 2.548e-05 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-204.34
    log_crmrte ~ prbarr + prbconv + prbpris + avgsen + log_polpc + 
        log_density + west + central + urban + pctmin80 + wcon + 
        wtuc + wtrd + wfir + wser + wmfg + wfed + wsta + wloc + log_mix + 
        log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - prbconv      1   0.00424 5.4686 -206.28  0.0520 0.8202508    
    - wmfg         1   0.01277 5.4771 -206.14  0.1566 0.6935552    
    - wcon         1   0.01321 5.4776 -206.13  0.1619 0.6886635    
    - wsta         1   0.02709 5.4914 -205.91  0.3321 0.5663453    
    - wloc         1   0.06061 5.5250 -205.36  0.7432 0.3917206    
    - wtuc         1   0.11252 5.5769 -204.53  1.3796 0.2443236    
    <none>                     5.4644 -204.34                      
    - pctmin80     1   0.12567 5.5900 -204.32  1.5409 0.2188121    
    - urban        1   0.16985 5.6342 -203.62  2.0825 0.1536514    
    - west         1   0.21741 5.6818 -202.87  2.6658 0.1072192    
    - prbpris      1   0.25483 5.7192 -202.29  3.1245 0.0816755 .  
    - wser         1   0.26137 5.7257 -202.19  3.2048 0.0779414 .  
    - wtrd         1   0.27772 5.7421 -201.93  3.4052 0.0694108 .  
    - central      1   0.29423 5.7586 -201.68  3.6077 0.0618199 .  
    - wfir         1   0.32034 5.7847 -201.27  3.9278 0.0515975 .  
    - log_mix      1   0.32796 5.7923 -201.16  4.0212 0.0489744 *  
    - log_pctymle  1   0.33797 5.8023 -201.00  4.1440 0.0457399 *  
    - wfed         1   0.34484 5.8092 -200.90  4.2282 0.0436563 *  
    - avgsen       1   0.55987 6.0242 -197.66  6.8647 0.0108657 *  
    - prbarr       1   0.78653 6.2509 -194.38  9.6438 0.0027843 ** 
    - log_density  1   1.06270 6.5270 -190.53 13.0300 0.0005862 ***
    - log_polpc    1   2.05589 7.5202 -177.92 25.2079 4.048e-06 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-206.28
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + urban + pctmin80 + wcon + wtuc + wtrd + 
        wfir + wser + wmfg + wfed + wsta + wloc + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - wcon         1   0.01122 5.4798 -208.09  0.1395 0.7099223    
    - wmfg         1   0.01467 5.4833 -208.04  0.1825 0.6706225    
    - wsta         1   0.02933 5.4979 -207.80  0.3647 0.5479222    
    - wloc         1   0.06325 5.5318 -207.25  0.7865 0.3783014    
    - wtuc         1   0.11736 5.5860 -206.39  1.4593 0.2312260    
    - pctmin80     1   0.12311 5.5917 -206.29  1.5309 0.2202371    
    <none>                     5.4686 -206.28                      
    - urban        1   0.18336 5.6520 -205.34  2.2800 0.1356856    
    - west         1   0.21729 5.6859 -204.81  2.7019 0.1048430    
    - prbpris      1   0.25973 5.7283 -204.15  3.2297 0.0767550 .  
    - wser         1   0.26735 5.7359 -204.03  3.3244 0.0726549 .  
    - central      1   0.29386 5.7625 -203.62  3.6541 0.0601477 .  
    - wtrd         1   0.31436 5.7830 -203.30  3.9089 0.0520859 .  
    - log_mix      1   0.32811 5.7967 -203.09  4.0799 0.0473388 *  
    - wfed         1   0.34313 5.8117 -202.86  4.2667 0.0426801 *  
    - wfir         1   0.35611 5.8247 -202.66  4.4281 0.0390510 *  
    - log_pctymle  1   0.35653 5.8251 -202.66  4.4333 0.0389398 *  
    - avgsen       1   0.57020 6.0388 -199.45  7.0903 0.0096659 ** 
    - prbarr       1   0.79348 6.2621 -196.22  9.8666 0.0024919 ** 
    - log_density  1   1.12302 6.5916 -191.65 13.9643 0.0003839 ***
    - log_polpc    1   2.57250 8.0411 -173.96 31.9882  3.36e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-208.09
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + urban + pctmin80 + wtuc + wtrd + wfir + 
        wser + wmfg + wfed + wsta + wloc + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - wmfg         1   0.01662 5.4964 -209.82  0.2092 0.6488146    
    - wsta         1   0.02537 5.5052 -209.68  0.3194 0.5737993    
    - wloc         1   0.08477 5.5646 -208.73  1.0674 0.3051474    
    - pctmin80     1   0.11608 5.5959 -208.23  1.4616 0.2307973    
    <none>                     5.4798 -208.09                      
    - wtuc         1   0.14269 5.6225 -207.81  1.7967 0.1845033    
    - urban        1   0.18997 5.6698 -207.06  2.3921 0.1265280    
    - west         1   0.23268 5.7125 -206.39  2.9299 0.0914459 .  
    - wser         1   0.25989 5.7397 -205.97  3.2725 0.0748075 .  
    - prbpris      1   0.28286 5.7627 -205.61  3.5617 0.0633352 .  
    - central      1   0.28628 5.7661 -205.56  3.6048 0.0617957 .  
    - log_mix      1   0.32887 5.8087 -204.91  4.1410 0.0457006 *  
    - wtrd         1   0.33096 5.8108 -204.87  4.1674 0.0450360 *  
    - wfir         1   0.34756 5.8274 -204.62  4.3764 0.0401220 *  
    - wfed         1   0.35590 5.8357 -204.49  4.4813 0.0378750 *  
    - log_pctymle  1   0.36200 5.8418 -204.40  4.5582 0.0363139 *  
    - avgsen       1   0.58376 6.0636 -201.08  7.3506 0.0084535 ** 
    - prbarr       1   0.78633 6.2661 -198.16  9.9011 0.0024386 ** 
    - log_density  1   1.12180 6.6016 -193.52 14.1253 0.0003542 ***
    - log_polpc    1   2.56816 8.0480 -175.89 32.3374 2.871e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-209.82
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + urban + pctmin80 + wtuc + wtrd + wfir + 
        wser + wfed + wsta + wloc + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - wsta         1   0.02922 5.5257 -211.35  0.3721 0.5438323    
    - wloc         1   0.08087 5.5773 -210.52  1.0299 0.3136786    
    - pctmin80     1   0.11749 5.6139 -209.94  1.4963 0.2253408    
    <none>                     5.4964 -209.82                      
    - wtuc         1   0.12733 5.6238 -209.79  1.6216 0.2070869    
    - urban        1   0.17569 5.6721 -209.02  2.2375 0.1391943    
    - west         1   0.22872 5.7252 -208.19  2.9129 0.0923064 .  
    - wser         1   0.27320 5.7696 -207.51  3.4794 0.0663282 .  
    - prbpris      1   0.27497 5.7714 -207.48  3.5018 0.0654798 .  
    - central      1   0.27927 5.7757 -207.41  3.5567 0.0634561 .  
    - wfed         1   0.34125 5.8377 -206.46  4.3461 0.0407443 *  
    - log_mix      1   0.34585 5.8423 -206.39  4.4046 0.0394507 *  
    - wtrd         1   0.35681 5.8532 -206.23  4.5441 0.0365439 *  
    - log_pctymle  1   0.36141 5.8578 -206.16  4.6028 0.0353904 *  
    - wfir         1   0.39481 5.8912 -205.65  5.0281 0.0281085 *  
    - avgsen       1   0.56716 6.0636 -203.08  7.2231 0.0089856 ** 
    - prbarr       1   0.77923 6.2757 -200.02  9.9239 0.0024001 ** 
    - log_density  1   1.11302 6.6094 -195.41 14.1748 0.0003434 ***
    - log_polpc    1   2.55669 8.0531 -177.83 32.5608 2.564e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-211.35
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + urban + pctmin80 + wtuc + wtrd + wfir + 
        wser + wfed + wloc + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - wloc         1   0.09943 5.6251 -211.76  1.2776 0.2621425    
    - wtuc         1   0.10670 5.6324 -211.65  1.3711 0.2455461    
    - pctmin80     1   0.12077 5.6464 -211.43  1.5517 0.2169729    
    <none>                     5.5257 -211.35                      
    - west         1   0.22652 5.7522 -209.78  2.9106 0.0923704 .  
    - urban        1   0.25457 5.7802 -209.34  3.2710 0.0747492 .  
    - prbpris      1   0.26007 5.7857 -209.26  3.3416 0.0717488 .  
    - central      1   0.26194 5.7876 -209.23  3.3657 0.0707568 .  
    - wser         1   0.30427 5.8299 -208.58  3.9096 0.0518943 .  
    - wtrd         1   0.32773 5.8534 -208.22  4.2110 0.0438505 *  
    - log_mix      1   0.32982 5.8555 -208.19  4.2379 0.0432015 *  
    - wfir         1   0.36599 5.8916 -207.64  4.7027 0.0334677 *  
    - wfed         1   0.40270 5.9284 -207.09  5.1744 0.0259455 *  
    - log_pctymle  1   0.45545 5.9811 -206.30  5.8521 0.0181229 *  
    - avgsen       1   0.53926 6.0649 -205.06  6.9291 0.0103979 *  
    - prbarr       1   0.79090 6.3165 -201.45 10.1624 0.0021324 ** 
    - log_density  1   1.09217 6.6178 -197.30 14.0335 0.0003623 ***
    - log_polpc    1   2.52845 8.0541 -179.82 32.4884 2.542e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-211.76
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + urban + pctmin80 + wtuc + wtrd + wfir + 
        wser + wfed + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - pctmin80     1   0.11191 5.7370 -212.01  1.4324 0.2352997    
    - wtuc         1   0.11264 5.7377 -212.00  1.4417 0.2337928    
    <none>                     5.6251 -211.76                      
    - wser         1   0.23888 5.8640 -210.06  3.0576 0.0846211 .  
    - central      1   0.24223 5.8673 -210.01  3.1005 0.0825168 .  
    - prbpris      1   0.25109 5.8762 -209.88  3.2139 0.0772140 .  
    - urban        1   0.26821 5.8933 -209.62  3.4330 0.0680023 .  
    - west         1   0.27022 5.8953 -209.59  3.4588 0.0670014 .  
    - log_mix      1   0.29691 5.9220 -209.19  3.8004 0.0551362 .  
    - wfir         1   0.31976 5.9448 -208.84  4.0929 0.0467786 *  
    - wfed         1   0.41408 6.0392 -207.44  5.3002 0.0242175 *  
    - wtrd         1   0.43919 6.0643 -207.07  5.6216 0.0204194 *  
    - log_pctymle  1   0.44927 6.0744 -206.93  5.7506 0.0190773 *  
    - avgsen       1   0.58476 6.2098 -204.96  7.4848 0.0078301 ** 
    - prbarr       1   0.80867 6.4338 -201.81 10.3508 0.0019405 ** 
    - log_density  1   1.04196 6.6670 -198.64 13.3369 0.0004903 ***
    - log_polpc    1   2.66801 8.2931 -179.22 34.1501 1.372e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-212.01
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + urban + wtuc + wtrd + wfir + wser + wfed + 
        log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - wtuc         1   0.12455 5.8615 -212.10  1.5849 0.2120709    
    <none>                     5.7370 -212.01                      
    - prbpris      1   0.18242 5.9194 -211.23  2.3212 0.1319411    
    - wfir         1   0.26725 6.0042 -209.96  3.4006 0.0692321 .  
    - wser         1   0.28833 6.0253 -209.65  3.6689 0.0593542 .  
    - log_mix      1   0.31272 6.0497 -209.29  3.9792 0.0498003 *  
    - urban        1   0.33967 6.0767 -208.89  4.3222 0.0411332 *  
    - wtrd         1   0.37125 6.1082 -208.43  4.7240 0.0329892 *  
    - log_pctymle  1   0.45458 6.1916 -207.22  5.7843 0.0187060 *  
    - central      1   0.51092 6.2479 -206.42  6.5012 0.0128792 *  
    - wfed         1   0.57233 6.3093 -205.55  7.2826 0.0086447 ** 
    - avgsen       1   0.66358 6.4006 -204.27  8.4437 0.0048460 ** 
    - prbarr       1   0.73151 6.4685 -203.33  9.3081 0.0031794 ** 
    - log_density  1   0.93170 6.6687 -200.62 11.8554 0.0009556 ***
    - west         1   1.30663 7.0436 -195.75 16.6261 0.0001147 ***
    - log_polpc    1   2.58603 8.3230 -180.90 32.9057 2.053e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-212.1
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + urban + wtrd + wfir + wser + wfed + log_mix + 
        log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    <none>                     5.8615 -212.10                      
    - prbpris      1   0.16653 6.0281 -211.61  2.1024 0.1512920    
    - wser         1   0.22865 6.0902 -210.69  2.8867 0.0935172 .  
    - wfir         1   0.28205 6.1436 -209.92  3.5607 0.0630842 .  
    - log_mix      1   0.29241 6.1540 -209.77  3.6916 0.0585376 .  
    - urban        1   0.32884 6.1904 -209.24  4.1514 0.0451752 *  
    - log_pctymle  1   0.38804 6.2496 -208.40  4.8988 0.0299577 *  
    - wtrd         1   0.39175 6.2533 -208.34  4.9457 0.0292065 *  
    - central      1   0.47324 6.3348 -207.19  5.9745 0.0168963 *  
    - avgsen       1   0.59231 6.4539 -205.53  7.4777 0.0078124 ** 
    - wfed         1   0.60061 6.4622 -205.42  7.5824 0.0074116 ** 
    - prbarr       1   0.72345 6.5850 -203.74  9.1333 0.0034463 ** 
    - log_density  1   1.11675 6.9783 -198.58 14.0986 0.0003431 ***
    - west         1   1.21723 7.0788 -197.31 15.3672 0.0001959 ***
    - log_polpc    1   2.65006 8.5116 -180.90 33.4561 1.635e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1



    
    Call:
    lm(formula = log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + 
        log_density + west + central + urban + wtrd + wfir + wser + 
        wfed + log_mix + log_pctymle, data = logcrime)
    
    Coefficients:
    (Intercept)       prbarr      prbpris       avgsen    log_polpc  log_density  
       2.559665    -1.082621    -0.607782    -0.036255     0.691947     0.119319  
           west      central        urban         wtrd         wfir         wser  
      -0.319797    -0.183172     0.261067     0.003125    -0.001585    -0.001769  
           wfed      log_mix  log_pctymle  
       0.002217     0.142843     0.372494  




```R
plot(lm(formula = log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + 
    log_density + west + central + urban + wtrd + wfir + wser + 
    wfed + log_mix + log_pctymle, data = logcrime))
```


![png](output_38_0.png)



![png](output_38_1.png)



![png](output_38_2.png)



![png](output_38_3.png)


We note that county 173  (point 78) has a large Cook's distance. It also has a population density near zero, and a log that is nearly 8 standard deviations below the mean. If the county is that thinly populated, all of its statistics are called into question because of the small sample sizes. Even a single additional crime/arrest could have a big effect on the crime rate/probability of arrest. For this reason, we remove this data point from the data set and rerun our backwards selection.


```R
m2 <- lm(log_crmrte ~ ., data = logcrime[-78,])
step(m2, direction="backward", test="F")
```

    Start:  AIC=-212.71
    log_crmrte ~ prbarr + prbconv + prbpris + avgsen + log_polpc + 
        log_density + log_taxpc + west + central + urban + pctmin80 + 
        wcon + wtuc + wtrd + wfir + wser + wmfg + wfed + wsta + wloc + 
        log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - prbconv      1   0.00000 4.6526 -214.71  0.0000  0.998268    
    - wmfg         1   0.00020 4.6528 -214.71  0.0028  0.957894    
    - wloc         1   0.01100 4.6636 -214.50  0.1537  0.696306    
    - urban        1   0.01482 4.6674 -214.43  0.2070  0.650644    
    - log_taxpc    1   0.03445 4.6870 -214.06  0.4812  0.490339    
    - wsta         1   0.03595 4.6885 -214.03  0.5023  0.481037    
    - wcon         1   0.03813 4.6907 -213.99  0.5327  0.468116    
    - prbpris      1   0.09293 4.7455 -212.97  1.2982  0.258715    
    <none>                     4.6526 -212.71                      
    - wtuc         1   0.11209 4.7647 -212.62  1.5659  0.215286    
    - wfed         1   0.12146 4.7740 -212.44  1.6969  0.197288    
    - west         1   0.14963 4.8022 -211.93  2.0904  0.153029    
    - wtrd         1   0.15504 4.8076 -211.83  2.1660  0.145914    
    - log_pctymle  1   0.17007 4.8226 -211.55  2.3760  0.128066    
    - avgsen       1   0.27589 4.9285 -209.64  3.8544  0.053897 .  
    - pctmin80     1   0.28642 4.9390 -209.46  4.0015  0.049641 *  
    - log_mix      1   0.31273 4.9653 -208.99  4.3691  0.040512 *  
    - wser         1   0.31427 4.9668 -208.96  4.3906  0.040038 *  
    - wfir         1   0.37938 5.0320 -207.81  5.3002  0.024535 *  
    - central      1   0.44635 5.0989 -206.65  6.2359  0.015061 *  
    - prbarr       1   0.71695 5.3695 -202.10 10.0163  0.002361 ** 
    - log_polpc    1   0.76211 5.4147 -201.36 10.6473  0.001759 ** 
    - log_density  1   1.54439 6.1970 -189.49 21.5762 1.706e-05 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-214.71
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        log_taxpc + west + central + urban + pctmin80 + wcon + wtuc + 
        wtrd + wfir + wser + wmfg + wfed + wsta + wloc + log_mix + 
        log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - wmfg         1   0.00020 4.6528 -216.71  0.0029  0.957418    
    - wloc         1   0.01103 4.6636 -216.50  0.1565  0.693671    
    - urban        1   0.01484 4.6674 -216.43  0.2105  0.647916    
    - log_taxpc    1   0.03612 4.6887 -216.03  0.5124  0.476608    
    - wsta         1   0.03627 4.6888 -216.03  0.5145  0.475710    
    - wcon         1   0.03933 4.6919 -215.97  0.5580  0.457730    
    - prbpris      1   0.09310 4.7457 -214.97  1.3206  0.254625    
    <none>                     4.6526 -214.71                      
    - wtuc         1   0.11282 4.7654 -214.60  1.6004  0.210298    
    - wfed         1   0.12317 4.7757 -214.41  1.7472  0.190788    
    - west         1   0.15062 4.8032 -213.91  2.1367  0.148559    
    - wtrd         1   0.16342 4.8160 -213.68  2.3183  0.132638    
    - log_pctymle  1   0.17731 4.8299 -213.42  2.5152  0.117533    
    - avgsen       1   0.27632 4.9289 -211.63  3.9198  0.051892 .  
    - pctmin80     1   0.28665 4.9392 -211.45  4.0663  0.047818 *  
    - log_mix      1   0.31324 4.9658 -210.98  4.4435  0.038836 *  
    - wser         1   0.31828 4.9709 -210.89  4.5150  0.037348 *  
    - wfir         1   0.39951 5.0521 -209.46  5.6674  0.020179 *  
    - central      1   0.44696 5.0995 -208.64  6.3404  0.014235 *  
    - prbarr       1   0.73901 5.3916 -203.74 10.4834  0.001886 ** 
    - log_polpc    1   0.79712 5.4497 -202.80 11.3077  0.001288 ** 
    - log_density  1   1.58124 6.2338 -190.97 22.4309 1.197e-05 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-216.71
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        log_taxpc + west + central + urban + pctmin80 + wcon + wtuc + 
        wtrd + wfir + wser + wfed + wsta + wloc + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - wloc         1   0.01086 4.6636 -218.50  0.1563  0.693830    
    - urban        1   0.01587 4.6687 -218.41  0.2286  0.634154    
    - log_taxpc    1   0.03597 4.6888 -218.03  0.5180  0.474201    
    - wsta         1   0.03710 4.6899 -218.01  0.5343  0.467368    
    - wcon         1   0.04021 4.6930 -217.95  0.5790  0.449355    
    - prbpris      1   0.09315 4.7459 -216.96  1.3414  0.250903    
    <none>                     4.6528 -216.71                      
    - wtuc         1   0.11783 4.7706 -216.51  1.6967  0.197178    
    - wfed         1   0.12654 4.7793 -216.35  1.8221  0.181606    
    - west         1   0.15066 4.8034 -215.90  2.1695  0.145454    
    - wtrd         1   0.16659 4.8194 -215.61  2.3989  0.126128    
    - log_pctymle  1   0.17740 4.8302 -215.42  2.5546  0.114679    
    - avgsen       1   0.28255 4.9353 -213.52  4.0688  0.047693 *  
    - pctmin80     1   0.28743 4.9402 -213.43  4.1390  0.045868 *  
    - log_mix      1   0.31552 4.9683 -212.94  4.5434  0.036717 *  
    - wser         1   0.32094 4.9737 -212.84  4.6216  0.035186 *  
    - wfir         1   0.42296 5.0757 -211.05  6.0907  0.016149 *  
    - central      1   0.44691 5.0997 -210.64  6.4356  0.013521 *  
    - prbarr       1   0.73967 5.3925 -205.73 10.6513  0.001733 ** 
    - log_polpc    1   0.79860 5.4514 -204.77 11.4998  0.001171 ** 
    - log_density  1   1.59542 6.2482 -192.76 22.9740 9.496e-06 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-218.5
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        log_taxpc + west + central + urban + pctmin80 + wcon + wtuc + 
        wtrd + wfir + wser + wfed + wsta + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - urban        1   0.01891 4.6825 -220.15  0.2757  0.601211    
    - log_taxpc    1   0.03815 4.7018 -219.79  0.5562  0.458354    
    - wsta         1   0.04635 4.7100 -219.63  0.6759  0.413889    
    - wcon         1   0.05682 4.7205 -219.44  0.8285  0.365917    
    - prbpris      1   0.08867 4.7523 -218.85  1.2930  0.259495    
    <none>                     4.6636 -218.50                      
    - wtuc         1   0.11790 4.7815 -218.31  1.7191  0.194215    
    - wfed         1   0.12356 4.7872 -218.20  1.8017  0.183973    
    - west         1   0.15698 4.8206 -217.59  2.2889  0.134937    
    - log_pctymle  1   0.17199 4.8356 -217.32  2.5078  0.117924    
    - wtrd         1   0.18915 4.8528 -217.00  2.7580  0.101376    
    - avgsen       1   0.28969 4.9533 -215.20  4.2239  0.043701 *  
    - pctmin80     1   0.29007 4.9537 -215.19  4.2294  0.043569 *  
    - log_mix      1   0.30812 4.9718 -214.87  4.4927  0.037693 *  
    - wser         1   0.31037 4.9740 -214.83  4.5254  0.037025 *  
    - wfir         1   0.41542 5.0790 -213.00  6.0571  0.016393 *  
    - central      1   0.44709 5.1107 -212.45  6.5190  0.012921 *  
    - prbarr       1   0.74682 5.4105 -207.43 10.8893  0.001542 ** 
    - log_polpc    1   0.80509 5.4687 -206.49 11.7389  0.001042 ** 
    - log_density  1   1.65067 6.3143 -193.84 24.0682 6.094e-06 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-220.15
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        log_taxpc + west + central + pctmin80 + wcon + wtuc + wtrd + 
        wfir + wser + wfed + wsta + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - log_taxpc    1   0.02278 4.7053 -221.72  0.3357  0.564192    
    - wsta         1   0.03433 4.7169 -221.50  0.5059  0.479327    
    - wcon         1   0.05462 4.7372 -221.13  0.8049  0.372760    
    - prbpris      1   0.09019 4.7727 -220.47  1.3290  0.252957    
    <none>                     4.6825 -220.15                      
    - wtuc         1   0.11702 4.7996 -219.97  1.7244  0.193481    
    - wfed         1   0.12630 4.8088 -219.81  1.8610  0.176938    
    - log_pctymle  1   0.17046 4.8530 -219.00  2.5119  0.117563    
    - wtrd         1   0.18230 4.8648 -218.79  2.6863  0.105764    
    - west         1   0.19810 4.8806 -218.50  2.9192  0.092027 .  
    - pctmin80     1   0.27120 4.9537 -217.19  3.9962  0.049543 *  
    - log_mix      1   0.29025 4.9728 -216.85  4.2770  0.042383 *  
    - wser         1   0.30829 4.9908 -216.54  4.5428  0.036622 *  
    - avgsen       1   0.31942 5.0020 -216.34  4.7069  0.033490 *  
    - wfir         1   0.40383 5.0864 -214.87  5.9507  0.017287 *  
    - central      1   0.44370 5.1262 -214.18  6.5382  0.012761 *  
    - prbarr       1   0.72880 5.4113 -209.42 10.7392  0.001644 ** 
    - log_polpc    1   0.91687 5.5994 -206.41 13.5107  0.000465 ***
    - log_density  1   2.14513 6.8277 -188.96 31.6097 3.708e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-221.72
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + pctmin80 + wcon + wtuc + wtrd + wfir + wser + 
        wfed + wsta + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - wsta         1   0.04163 4.7470 -222.94  0.6193  0.433981    
    - wcon         1   0.06249 4.7678 -222.56  0.9297  0.338264    
    - prbpris      1   0.09476 4.8001 -221.97  1.4097  0.239128    
    - wfed         1   0.10396 4.8093 -221.80  1.5465  0.217799    
    <none>                     4.7053 -221.72                      
    - wtuc         1   0.12653 4.8319 -221.38  1.8824  0.174440    
    - log_pctymle  1   0.14769 4.8530 -221.00  2.1971  0.142761    
    - wtrd         1   0.20313 4.9085 -220.00  3.0219  0.086546 .  
    - west         1   0.25951 4.9648 -219.00  3.8607  0.053401 .  
    - pctmin80     1   0.26351 4.9688 -218.93  3.9202  0.051643 .  
    - log_mix      1   0.27439 4.9797 -218.73  4.0821  0.047168 *  
    - wser         1   0.28597 4.9913 -218.53  4.2543  0.042864 *  
    - avgsen       1   0.33835 5.0437 -217.61  5.0336  0.028027 *  
    - wfir         1   0.41720 5.1225 -216.25  6.2066  0.015100 *  
    - central      1   0.50113 5.2065 -214.81  7.4552  0.007996 ** 
    - prbarr       1   0.71706 5.4224 -211.24 10.6676  0.001690 ** 
    - log_polpc    1   1.64016 6.3455 -197.40 24.4003 5.130e-06 ***
    - log_density  1   2.14278 6.8481 -190.70 31.8776 3.262e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-222.95
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + pctmin80 + wcon + wtuc + wtrd + wfir + wser + 
        wfed + log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - wcon         1   0.05734 4.8043 -223.89  0.8576  0.357544    
    - prbpris      1   0.08386 4.8308 -223.40  1.2543  0.266506    
    - wtuc         1   0.10422 4.8512 -223.03  1.5588  0.215947    
    <none>                     4.7470 -222.94                      
    - wfed         1   0.12884 4.8758 -222.59  1.9271  0.169418    
    - wtrd         1   0.17557 4.9225 -221.75  2.6260  0.109562    
    - log_pctymle  1   0.19601 4.9430 -221.38  2.9317  0.091219 .  
    - west         1   0.25607 5.0030 -220.32  3.8300  0.054274 .  
    - log_mix      1   0.25891 5.0059 -220.27  3.8725  0.052988 .  
    - pctmin80     1   0.28335 5.0303 -219.84  4.2381  0.043196 *  
    - wser         1   0.30200 5.0490 -219.52  4.5171  0.037039 *  
    - avgsen       1   0.30519 5.0521 -219.46  4.5647  0.036084 *  
    - wfir         1   0.37755 5.1245 -218.21  5.6469  0.020186 *  
    - central      1   0.47701 5.2240 -216.52  7.1346  0.009369 ** 
    - prbarr       1   0.74258 5.4895 -212.16 11.1067  0.001369 ** 
    - log_polpc    1   1.60361 6.3506 -199.33 23.9851 5.894e-06 ***
    - log_density  1   2.20004 6.9470 -191.43 32.9058 2.195e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-223.89
    log_crmrte ~ prbarr + prbpris + avgsen + log_polpc + log_density + 
        west + central + pctmin80 + wtuc + wtrd + wfir + wser + wfed + 
        log_mix + log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    - prbpris      1   0.10961 4.9139 -223.90  1.6427  0.204065    
    <none>                     4.8043 -223.89                      
    - wfed         1   0.14094 4.9452 -223.34  2.1121  0.150479    
    - wtuc         1   0.15538 4.9597 -223.09  2.3286  0.131400    
    - log_pctymle  1   0.19903 5.0033 -222.32  2.9827  0.088445 .  
    - wtrd         1   0.23451 5.0388 -221.69  3.5145  0.064891 .  
    - pctmin80     1   0.25577 5.0601 -221.32  3.8331  0.054124 .  
    - log_mix      1   0.26042 5.0647 -221.24  3.9028  0.052039 .  
    - wser         1   0.26709 5.0714 -221.13  4.0028  0.049196 *  
    - west         1   0.29688 5.1012 -220.61  4.4492  0.038395 *  
    - avgsen       1   0.33700 5.1413 -219.92  5.0505  0.027684 *  
    - wfir         1   0.35490 5.1592 -219.62  5.3188  0.023979 *  
    - central      1   0.44927 5.2536 -218.02  6.7330  0.011462 *  
    - prbarr       1   0.72924 5.5335 -213.45 10.9289  0.001478 ** 
    - log_polpc    1   1.63181 6.4361 -200.16 24.4552 4.810e-06 ***
    - log_density  1   2.17157 6.9759 -193.07 32.5445 2.411e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Step:  AIC=-223.9
    log_crmrte ~ prbarr + avgsen + log_polpc + log_density + west + 
        central + pctmin80 + wtuc + wtrd + wfir + wser + wfed + log_mix + 
        log_pctymle
    
                  Df Sum of Sq    RSS     AIC F value    Pr(>F)    
    <none>                     4.9139 -223.90                      
    - wtuc         1   0.14834 5.0622 -223.29  2.2037  0.141983    
    - wfed         1   0.15041 5.0643 -223.25  2.2345  0.139268    
    - wtrd         1   0.19011 5.1040 -222.56  2.8242  0.097130 .  
    - pctmin80     1   0.19322 5.1071 -222.51  2.8705  0.094480 .  
    - log_pctymle  1   0.19582 5.1097 -222.47  2.9091  0.092333 .  
    - log_mix      1   0.22095 5.1349 -222.03  3.2824  0.074138 .  
    - wser         1   0.23375 5.1477 -221.81  3.4725  0.066418 .  
    - avgsen       1   0.27988 5.1938 -221.03  4.1579  0.045062 *  
    - wfir         1   0.32334 5.2372 -220.30  4.8035  0.031592 *  
    - west         1   0.42866 5.3426 -218.54  6.3680  0.013796 *  
    - central      1   0.58363 5.4975 -216.03  8.6703  0.004336 ** 
    - prbarr       1   0.68358 5.5975 -214.44 10.1551  0.002119 ** 
    - log_polpc    1   1.55506 6.4690 -201.71 23.1017 8.002e-06 ***
    - log_density  1   2.16607 7.0800 -193.77 32.1787 2.658e-07 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1



    
    Call:
    lm(formula = log_crmrte ~ prbarr + avgsen + log_polpc + log_density + 
        west + central + pctmin80 + wtuc + wtrd + wfir + wser + wfed + 
        log_mix + log_pctymle, data = logcrime[-78, ])
    
    Coefficients:
    (Intercept)       prbarr       avgsen    log_polpc  log_density         west  
      1.3665009   -1.0588956   -0.0252071    0.5563606    0.3634625   -0.2740753  
        central     pctmin80         wtuc         wtrd         wfir         wser  
     -0.2263759    0.0044536    0.0006486    0.0022252   -0.0017163   -0.0018229  
           wfed      log_mix  log_pctymle  
      0.0011828    0.1228953    0.2748621  




```R
plot(lm(formula = log_crmrte ~ prbarr + avgsen + log_polpc + log_density + 
    west + central + pctmin80 + wtuc + wtrd + wfir + wser + wfed + 
    log_mix + log_pctymle, data = logcrime[-78, ]))
```


![png](output_41_0.png)



![png](output_41_1.png)



![png](output_41_2.png)



![png](output_41_3.png)


Removing point 78 has ensured that no point has too much leverage. We will perform further analysis of these plots below.

AIC is one method that can be used to balance minimizing R-squared with providing a penalty for adding a new variable. I.e., lower AIC indicates less information lost without overfitting. As we can see, this final model minimizes AIC. 
Below is our final model.


```R
mfinal <- lm(formula = log_crmrte ~ prbarr + avgsen + log_polpc + log_density + 
    west + central + pctmin80 + wtuc + wtrd + wfir + wser + wfed + 
    log_mix + log_pctymle, data = logcrime[-78, ])
summary(mfinal)
```


    
    Call:
    lm(formula = log_crmrte ~ prbarr + avgsen + log_polpc + log_density + 
        west + central + pctmin80 + wtuc + wtrd + wfir + wser + wfed + 
        log_mix + log_pctymle, data = logcrime[-78, ])
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -0.83355 -0.11522  0.02452  0.11142  0.75247 
    
    Coefficients:
                  Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  1.3665009  0.8928799   1.530  0.13023    
    prbarr      -1.0588956  0.3322860  -3.187  0.00212 ** 
    avgsen      -0.0252071  0.0123619  -2.039  0.04506 *  
    log_polpc    0.5563606  0.1157536   4.806 8.00e-06 ***
    log_density  0.3634625  0.0640730   5.673 2.66e-07 ***
    west        -0.2740753  0.1086094  -2.523  0.01380 *  
    central     -0.2263759  0.0768799  -2.945  0.00434 ** 
    pctmin80     0.0044536  0.0026286   1.694  0.09448 .  
    wtuc         0.0006486  0.0004369   1.484  0.14198    
    wtrd         0.0022252  0.0013241   1.681  0.09713 .  
    wfir        -0.0017163  0.0007831  -2.192  0.03159 *  
    wser        -0.0018229  0.0009782  -1.863  0.06642 .  
    wfed         0.0011828  0.0007912   1.495  0.13927    
    log_mix      0.1228953  0.0678325   1.812  0.07414 .  
    log_pctymle  0.2748621  0.1611518   1.706  0.09233 .  
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 0.2594 on 73 degrees of freedom
    Multiple R-squared:  0.7906,	Adjusted R-squared:  0.7504 
    F-statistic: 19.69 on 14 and 73 DF,  p-value: < 2.2e-16




```R
coeftest(mfinal, vcov = vcovHC(mfinal, type = "HC0")) #heteroskedastic robust analysis
```


    
    t test of coefficients:
    
                   Estimate  Std. Error t value  Pr(>|t|)    
    (Intercept)  1.36650088  1.46659430  0.9318 0.3545365    
    prbarr      -1.05889560  0.26442948 -4.0045 0.0001479 ***
    avgsen      -0.02520711  0.00947781 -2.6596 0.0096108 ** 
    log_polpc    0.55636057  0.22945486  2.4247 0.0177962 *  
    log_density  0.36346248  0.04802061  7.5689 9.065e-11 ***
    west        -0.27407532  0.13684212 -2.0029 0.0489063 *  
    central     -0.22637590  0.08425811 -2.6867 0.0089301 ** 
    pctmin80     0.00445359  0.00345771  1.2880 0.2018082    
    wtuc         0.00064859  0.00049610  1.3074 0.1951828    
    wtrd         0.00222521  0.00115320  1.9296 0.0575441 .  
    wfir        -0.00171633  0.00067764 -2.5328 0.0134645 *  
    wser        -0.00182288  0.00079227 -2.3008 0.0242601 *  
    wfed         0.00118276  0.00081938  1.4435 0.1531644    
    log_mix      0.12289527  0.07147978  1.7193 0.0897974 .  
    log_pctymle  0.27486207  0.14609327  1.8814 0.0639028 .  
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1



Our final model contains the following variables: prbarr (probability of arrest), avgsen (avergage sentence length), log_polpc (log of police per capita), log_density (log of population density), west (indicator for county in the west), central (indicator for county in the center of the state), pctmin80 (percent minority), wtuc (transportation/utility wage), wtrd (wholesale wage), wfir (finance wage), wser (service wage), wfed (federal employee wage), log_mix (log of the crime mix). We believe that our explanatory variables are probability of arrests and average sentence length. We are controlling for police per capita, density, whether the county is in the west or central, the various wages, and mix of the crimes.

This is in line with our understanding of the variables. Having a higher probability of arrest will naturally deter criminals from committing crimes in the first place from the fear of getting caught and serving time. On top of that, increasing the penalty of crimes (having longer sentences) will deter criminals from committing crimes. Police per capita is controlled for to atone for the fact that having more police increases the rate of crime, since there are more officers to detect crime. Density is controlled for as well, since denser counties have more people, making crime more probable. West and central are controlled for to remove any locational biases that may arise. The wage data variables are indicative of the salary of the county, since wages are based on the cost of living of the specific city/area they reside in. We control for the mix of crime, because a higher mix of crime means that there are more face to face crimes. The police may be more inclined to go after face to face crimes as opposed to petty thefts, for example, so we are controlling for that effect to ensure that our explanatory variables are not related to the error term. We control for the percentage of young males, because that is the most commonly implicated demographic in crime. We also control for the percent minority because of the possibility of correlation between minority status and the probability of arrest, regardless of whether crimes were committed.

## Statistical Significance
To determine significance, we use the heteroskedastic robust tools from above for reasons stated in our CLM assumptions below.

The null hypothesis for each term in the regression equation is that the coefficient has no effect on the dependent variable (the coefficient equals 0). Looking at our model 2, we can see that our explanatory variables, probability of arrest and average sentence, have very low p-values of 0.0001479 and 0.0096108 respectively, indicating high statistical significance. We therefore reject the null hypothesis that the coefficients are equal to 0, making them more likely to be associated with our response variable. Analyzing the variables that we control for, police per capita and density both have low p-values of 0.0177962 and essentially 0 respectively, indicating very high statistical significance. We, therefore, reject the null hypothesis, making them more likely to be associated with our response variable. Similarly, we reject the null hypothesis for west, central, and wfir which all have statistically significant p-values of 0.0489063, 0.0089301, and 0.0134645. Wtuc, pctmin80, wtrd, wfed, mix, and pctymle all have p-values greater than 0.05, meaning we fail to reject the null hypothesis. However, we believe these variables are meaningful additions to the model, because they capture relationships between explanatory variables and control variables that may otherwise confound the regression.

Therefore, model 1 will contain only the explanatory variables, model 2 will contain the explanatory variables plus the variables we control for, and the final model contains all variables. 


```R
model1 <- lm(log_crmrte ~ prbarr + avgsen, data = logcrime[-78,])
model2 <- mfinal
model3 <- lm(log_crmrte ~ ., data = logcrime[-78,])
stargazer(model1, model2, model3, title="Summary of Models", align=TRUE, type = "text")
AIC(model1, model2, model3)
```

    
    Summary of Models
    =========================================================================================
                                                 Dependent variable:                         
                        ---------------------------------------------------------------------
                                                     log_crmrte                              
                                 (1)                    (2)                     (3)          
    -----------------------------------------------------------------------------------------
    prbarr                    -1.665***              -1.059***               -1.121***       
                               (0.497)                (0.332)                 (0.354)        
                                                                                             
    prbconv                                                                   0.0005         
                                                                              (0.227)        
                                                                                             
    prbpris                                                                   -0.493         
                                                                              (0.433)        
                                                                                             
    avgsen                      0.011                -0.025**                 -0.027*        
                               (0.020)                (0.012)                 (0.014)        
                                                                                             
    log_polpc                                        0.556***                0.511***        
                                                      (0.116)                 (0.157)        
                                                                                             
    log_density                                      0.363***                0.382***        
                                                      (0.064)                 (0.082)        
                                                                                             
    log_taxpc                                                                  0.122         
                                                                              (0.176)        
                                                                                             
    west                                             -0.274**                 -0.184         
                                                      (0.109)                 (0.127)        
                                                                                             
    central                                          -0.226***               -0.209**        
                                                      (0.077)                 (0.084)        
                                                                                             
    urban                                                                     -0.076         
                                                                              (0.167)        
                                                                                             
    pctmin80                                          0.004*                  0.006**        
                                                      (0.003)                 (0.003)        
                                                                                             
    wcon                                                                       0.001         
                                                                              (0.001)        
                                                                                             
    wtuc                                               0.001                   0.001         
                                                     (0.0004)                (0.0005)        
                                                                                             
    wtrd                                              0.002*                   0.002         
                                                      (0.001)                 (0.002)        
                                                                                             
    wfir                                             -0.002**                -0.002**        
                                                      (0.001)                 (0.001)        
                                                                                             
    wser                                              -0.002*                -0.002**        
                                                      (0.001)                 (0.001)        
                                                                                             
    wmfg                                                                     -0.00002        
                                                                             (0.0005)        
                                                                                             
    wfed                                               0.001                   0.001         
                                                      (0.001)                 (0.001)        
                                                                                             
    wsta                                                                       0.001         
                                                                              (0.001)        
                                                                                             
    wloc                                                                       0.001         
                                                                              (0.002)        
                                                                                             
    log_mix                                           0.123*                  0.153**        
                                                      (0.068)                 (0.073)        
                                                                                             
    log_pctymle                                       0.275*                   0.291         
                                                      (0.161)                 (0.189)        
                                                                                             
    Constant                  -3.144***                1.367                   0.544         
                               (0.256)                (0.893)                 (1.509)        
                                                                                             
    -----------------------------------------------------------------------------------------
    Observations                 88                     88                      88           
    R2                          0.122                  0.791                   0.802         
    Adjusted R2                 0.102                  0.750                   0.735         
    Residual Std. Error    0.492 (df = 85)        0.259 (df = 73)         0.268 (df = 65)    
    F Statistic         5.916*** (df = 2; 85) 19.686*** (df = 14; 73) 11.947*** (df = 22; 65)
    =========================================================================================
    Note:                                                         *p<0.1; **p<0.05; ***p<0.01



<table>
<caption>A data.frame: 3 × 2</caption>
<thead>
	<tr><th></th><th scope=col>df</th><th scope=col>AIC</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>model1</th><td> 4</td><td>129.94713</td></tr>
	<tr><th scope=row>model2</th><td>16</td><td> 27.82962</td></tr>
	<tr><th scope=row>model3</th><td>24</td><td> 39.02059</td></tr>
</tbody>
</table>



By design, our second model shows the strongest F-statistic and the lowest AIC (which makes sense, because these are the critera we used to select it). 
- The significant F-statistic indicates that each of our models are more informative than the intercept-only model. Here, model 2 has the strongest significance for the F-statistic. 
- A small AIC means that we have balanced retained information with overfitting. Here, model 2 has the best balance.
- Model 2 has the highest adjusted R-squared, another indicator that we have balanced retained information with overfitting. 

From model 1 to model 2, as we include the variables we are controlling for, the probability of arrest coefficient increases, indicating that we are accounting for relationships omitted from model 1. In model 2 to model 3, the coefficient decreases, the F-statistic and adjusted R-squared decreases, and AIC increases indicating that model 3 was overfitted. 

We expected a negative relationship between average sentence and crime rate; however, in model 1, it was positive and not significant. After including our control variables, the relationship became negative and significant, meaning we accounted for an omitted relationship that was masking the effect of average sentence. Then from model 2 to model 3, the coefficient did not change very much, indicating that model 2 is robust to the effect of adding variables.

The above leads us to conclude that model 2 is the best model. 

## Omitted Variables

1. We are controlling for police per capita since it is correlated with crime rate. But, we are introducing a bias where more police naturally increases the crime rate (since more police means more people who must report crimes, and there is more general promotion to report crimes). We think this push to report crimes increases the crime rate. 

2. We do not have data for unreported crime. If we truly knew about every crime that was commited, that would change the outcome of our model. If we knew about all unreported crimes, it would increase our crime rate. We also believe that these unreported crimes are of a different nature. For example, someone is more likely to report a murder than a pickpocket. This could change the way our variables interact with crime rate. 

3. Police effectiveness is not directly measured. For example, police may be less inclined to work in low income areas where the crime rate has a reputation for being higher. This could result in dissatisfaction of the employed officers, which may lead to a decline in the quality of work the officer does. Perpetrators might be more inclined to commit more crimes with less vigilance of officers. This would have a negative impact because as quality of police increases, the crime rate decreases. 

4. In counties with high police presence the crime rate tends to be higher. This could be because the added police presence encourages people to report crimes. However, in counties with low police presence, people may feel they will not be helped anyways, so do not report crimes.  We think this desire to report crimes increases the crime rate. 

5. Environmental variables such as weather can affect crime rates, but we do not have climate data. Hotter counties will have higher crime rates. 

6. We do not know the ratios of wages in an area. Although wage is included in this dataset, we cannot assume that the wages are equally spread (i.e. it is not accurate to assume that every county has the same ratio of people in finance, retail, etc.). We believe that areas with lower overall wage probably have higher crime rates. 

7. We do not have unemployement rate data. We think that higher unemployement would lead to a higher rate of petty crime or theft. 

8. We do not have any data about education levels of people in counties. Although this can be correlated with wage, we believe that an increase in education leads to a decrease in crime rate, even when holding wage constant.

Even with these omitted variables, we believe that probability of arrest and average sentence are still valid variables, because probability of arrest and average sentence are fairly independent from the listed omitted variables. If an effect was omitted, it should be fairly slight when it comes to affecting the coefficients of these variables. However, there are two variables that we think may have a slight effect on probability of arrest:

1. If all unreported crimes were reported, but not arrested, that would reduce the crime rate. However, it is reasonable to think that more reported crimes would lead to more arrests. It is hard to say how exactly this would affect probability of arrest. For the purpose of this report, we will assume the affect is negligible. 

2. The police per capita may effect probability of arrest. To check this, we will look at covariance.



```R
cov(logcrime$prbarr, logcrime$log_polpc)
cor(logcrime$prbarr, logcrime$log_polpc)
```


-0.00476903336736013



-0.134123063772953


Probability of arrest and police per capita have a low covariance and correlation. This implies that they have little relationship. Therefore, an increase in police per capita does not imply an increase in probability of arrest. With this information, we think that probability of arrest is a valid explanitory variable.


### Practical Significance


```R
cohen.d(logcrime$log_crmrte, logcrime$prbarr, na.rm = TRUE)
cohen.d(logcrime$log_crmrte, logcrime$avgsen, na.rm = TRUE)
```


    
    Cohen's d
    
    d estimate: -10.09228 (large)
    95 percent confidence interval:
         lower      upper 
    -11.188577  -8.995984 



    
    Cohen's d
    
    d estimate: -6.993916 (large)
    95 percent confidence interval:
        lower     upper 
    -7.783017 -6.204815 


In model 2 (the model we are choosing to use), increasing the probablity of arrest by 1 percent will decrease the crime rate by 1.059 percent. This is statstically significant with a p-value <.01.

Increasing the average sentence decreases the crime rate by 2.5 percent.  This is statistically significant with a p-value <.05.

We have tested for practical significance of the prbarr and avgsen on crmrte using Cohen's d. Our Cohen's d is quite large for both of them, indicating there is large practical significance here.

### Standardized coefficients
It would be interesting to know the relative effects of coefficients used in the model. That is, after accounting for the natural variability (variance) of each variable, which are the largest coefficients? To answer this, we standardize the cofficients using the beta function.


```R
lm.beta(mfinal)
```


<dl class=dl-horizontal>
	<dt>prbarr</dt>
		<dd>-0.217099916707094</dd>
	<dt>avgsen</dt>
		<dd>-0.125715934310937</dd>
	<dt>log_polpc</dt>
		<dd>0.341218294987609</dd>
	<dt>log_density</dt>
		<dd>0.534596134373808</dd>
	<dt>west</dt>
		<dd>-0.222421667509868</dd>
	<dt>central</dt>
		<dd>-0.213453972868746</dd>
	<dt>pctmin80</dt>
		<dd>0.145570939741538</dd>
	<dt>wtuc</dt>
		<dd>0.0932453720089866</dd>
	<dt>wtrd</dt>
		<dd>0.145818768011775</dd>
	<dt>wfir</dt>
		<dd>-0.179298111336099</dd>
	<dt>wser</dt>
		<dd>-0.155020317845523</dd>
	<dt>wfed</dt>
		<dd>0.135513185968296</dd>
	<dt>log_mix</dt>
		<dd>0.126622541661119</dd>
	<dt>log_pctymle</dt>
		<dd>0.105838923146368</dd>
</dl>



The population density (log_density) has the greatest effect, with a one standard deviation increase in density leading to over a half standard deviation increase in crime. This is not surprising, given that greater density means people would interact more and that there would be more opportunity for crime. This is one of the reasons we included it as a control variable. Police per capita (polpc), another control variable, has the second greatest effect, with a one standard deviation increase of police leading to about a third of a standard deviation increase in crime. The value for probability of arrest (prbarr) is among the higher valued standardized coefficients, accounting for about a twenty percent decrease in crime in terms of standard deviations. We consider it our primary explanitory variable. Our other explanitory variable, the average sentence length (avgsen), has a lower effect, accounting for about a twelve percent decrease in crime in terms of standard deviations. However, this is higher than the beta value of some of the control variables in the model, so we still consider its impact on crime to be important.

## CLM Assumptions

Below is our final model. We will test for the 6 assumptions of CLM. 


```R
summary(mfinal)
```


    
    Call:
    lm(formula = log_crmrte ~ prbarr + avgsen + log_polpc + log_density + 
        west + central + pctmin80 + wtuc + wtrd + wfir + wser + wfed + 
        log_mix + log_pctymle, data = logcrime[-78, ])
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -0.83355 -0.11522  0.02452  0.11142  0.75247 
    
    Coefficients:
                  Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  1.3665009  0.8928799   1.530  0.13023    
    prbarr      -1.0588956  0.3322860  -3.187  0.00212 ** 
    avgsen      -0.0252071  0.0123619  -2.039  0.04506 *  
    log_polpc    0.5563606  0.1157536   4.806 8.00e-06 ***
    log_density  0.3634625  0.0640730   5.673 2.66e-07 ***
    west        -0.2740753  0.1086094  -2.523  0.01380 *  
    central     -0.2263759  0.0768799  -2.945  0.00434 ** 
    pctmin80     0.0044536  0.0026286   1.694  0.09448 .  
    wtuc         0.0006486  0.0004369   1.484  0.14198    
    wtrd         0.0022252  0.0013241   1.681  0.09713 .  
    wfir        -0.0017163  0.0007831  -2.192  0.03159 *  
    wser        -0.0018229  0.0009782  -1.863  0.06642 .  
    wfed         0.0011828  0.0007912   1.495  0.13927    
    log_mix      0.1228953  0.0678325   1.812  0.07414 .  
    log_pctymle  0.2748621  0.1611518   1.706  0.09233 .  
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 0.2594 on 73 degrees of freedom
    Multiple R-squared:  0.7906,	Adjusted R-squared:  0.7504 
    F-statistic: 19.69 on 14 and 73 DF,  p-value: < 2.2e-16



#### 1: Linearity

There is no way to check for linearity because we can't look at our population. However, there is no need to check for linearity because we have not yet constrained our error term. Therefore, our model can be linear no matter what as long as we are okay with a poorly behaved error term. Thus, it is okay to assume linearity. 

#### 2: Random Sampling
To test for random sampling, we must know how our data is collected. Here, we know that each data point represents a county, but how were the counties selected? For example, did we pull every county from the state, did we choose convenient counties, or is it a truly random selection from the state? If the data is taken from a random selection of counties, then we can assume random sampling. Because we don't know that this is the case, we can't assume that this data is represenative of all counites. 

If we cannot assume random sampling, we may have issues with clustering. For example, state policies may have an impact on crime rate. If we knew this were the case, we would still know that the coefficients are unbiased, though less precise. 

This is not time-series data which leads us to believe that the data is not autocorrelated. It is possible that one county may have effects on nearby counties, and we don't know if these counties are neighbors or not. For the sake of this assignment, we will assume that the counties were randomly sampled and therefore not autocorrelated.

#### 3: Multicollinearity
When we run the model, if we have variables that are perfectly collinear, R will throw an error. In this case, there is no error when we run our model, so it is safe to assume that our model has no perfect collinearity. Also, the scatterplots revealed no perfect linear relationships.

#### 4: Zero-Conditional Mean
Looking at the residual vs. fitted plot, we see that the red line is flat enough around zero, indicating a zero-conditional mean. 


```R
plot(mfinal, main="Crime Rate")
```


![png](output_60_0.png)



![png](output_60_1.png)



![png](output_60_2.png)



![png](output_60_3.png)


#### 5: Homoskedasticity

In a homeskedatic model, our errors should be uniform around the red line in the residual vs. fitted plot, which appears to be the case. We can confirm homoskedasticity by showing that the results of the Breush-Pagan Test are not significant (p > .05).


```R
bptest(mfinal)
```


    
    	studentized Breusch-Pagan test
    
    data:  mfinal
    BP = 39.406, df = 14, p-value = 0.0003156



We can see that our p-value is less than .05. This indicates heteroskedasticity. This could be due to actual heteroskedasticity, or to large sample size (this test tends to come up significant for large samples). To be certain, we used heteroskedastic robust tools, which work for homoskedastic models; they are just more conservative. As shown above, the robust t-test results show significance of the coefficients of our explanitory variables.

#### 6: Normality of Error Terms

We can see from the Normal QQ plot above that the values mostly tend to follow the normal line, but there may be some non-normality at the extreme values. However, even if the extreme values are non-normal, we can rely on the asymptotic properties of OLS because we have a large sample size. Therefore, we will assume that our error term is normal.

## Policy Suggestion

Our models indicate a negative relationship between the probability of arrest and the length of sentences. Presumably, in counties where arrest rates are higher and sentences are longer, people are less inclined to commit crimes. Therefore, our policy suggestions revolve around increasing the arrest rate and length of sentences, or at least increasing the *perceived* values of these variables.

There are three ways to go about doing this:
1. The most obvious way to increase the number of arrests is to increase the number of police. However, our model indicates that there is a positive relationship between the police per capita and the crime rate. There is also little correlation between police per capita and probability of arrest. Therefore, we must be wary of simply increasing the number of police.
2. Better training and tools for police can improve their efficiency in making arrests without actually increasing the number of police. This could come in the form of subsidized tuition for criminology courses or better gear such as body cams, for example. This can be costly, but beneficial.
3. Since the reduction in crime appears to be related to the perception of bad outcomes from crime, our proposal is to increase the perceived arrest rate and sentence length. This can be a low-cost solution.

#### How can we increase the perceived arrest rate?
- When someone is arrested, make it public: post it in newspapers and on local news stations.
- Intentionally take data out of context: "We arrested 50 people this month for x crime." That may only be 10%, but it can sound like a lot to someone who is thinking of commiting a crime.

#### How can we increase the perception of average sentence lengths?
- Publicize the average sentence length for each crime
- Publicize convictions that have a long sentence
- Highlight the pain and expense involved in going through the criminal justice system

We believe that by increasing the perceived drawbacks of committing a crime, we will reduce the overall crime rate.
