# Lasso and Bootstrap Examples

### mtcars Analysis - Lasso Example
For response $$y$$ with predictors $$x_{1},...,x_{p}$$ the least squares estimator is the set of $$\beta$$s ,$$\left(\hat{\beta_{0}}, \hat{\beta_{1}}, ..., \hat{\beta_{p}} \right)$$, that minimizes

$$
\frac{1}{N}\sum_{i=1}^{n} \left( y_{i} - \beta_{0} - \beta_{1}x_{1} - ... - \beta_{p}x_{ip} \right)^2
$$

The lasso estimator, $$\lambda$$, is defined the same way as the least squared estimator, but it adds a penalty based on the value of lambda. This penalty will shrink the coefficients towards 0, creating a model with fewer predictors. This is especially helpful when the number of variables (p) is almost as big or bigger than the number of observations (n). 


```R
install.packages("glmnet")
library(glmnet)
```

    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpIsfd4l/downloaded_packages



```R
x <- with(mtcars, cbind(cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb))
y <- mtcars$mpg
set.seed(1)
lasso_m <- cv.glmnet(x,y)
coefficients(lasso_m, s='lambda.min')
```


    11 x 1 sparse Matrix of class "dgCMatrix"
                          1
    (Intercept) 36.44500429
    cyl         -0.89288058
    disp         .         
    hp          -0.01281976
    drat         .         
    wt          -2.78332595
    qsec         .         
    vs           .         
    am           0.01347181
    gear         .         
    carb         .         


I will use the variables cyl, hp, and wt in my model.

$$\lambda$$ was selected using 10 folds cross validation with a set seed of 1. Many different values of lambda were fit, then 10 folds cross validation was used on the lambda values to determine which lambda provided the smallest cross validation error, ($\lambda$ min). Then, the lambda that produced the smallest model within one standard error of the lambda min model was selected ($$\lambda$$ 1se).

The point of using the lasso model is to minimize the magnitude of coefficients. Some variables will have coefficients that start off large, then shrink to zero quickly through the lasso model, where others may start off with small coefficients but remain robust through the lasso model. This is because the size of the coefficient is related to the scale of the predictor. 

### Ornstein Car Dataset Analysis - Bootstrap Example

##### How bootstrapping works:
Step 1: Resample the data with replacement to get a new bootstrap data the same size as the original sample.

Step 2: Fit the linear regression model using the bootstrapped data.

Step 3: Repeat the above two steps 10,000 times. The standard errors for the intercept and asset coefficient are the standard deviation of the 10,000 intercept and asset coefficients, respectively.


```R
library(car)
```


```R
set.seed(1)
source("https://sites.google.com/site/bsherwood/bootstrap_code.r")
m1 <- lm(interlocks ~ assets, Ornstein)
bootstrap_lm(m1) # This function code can be found at the site inside the source function
```


<dl class=dl-horizontal>
	<dt>(Intercept)</dt>
		<dd>0.729621717413012</dd>
	<dt>assets</dt>
		<dd>8.84035418548277e-05</dd>
</dl>



##### Hypothesis test

Is the coefficient for assets zero, or not?

$$H_{0}: \beta_{1} = 0$$

$$H_{1}: \beta_{1} \neq 0$$

$$p-value: 8.840 \times 10^{-5}$$

This is a very small p-value, so we can reject our null hypothesis and assume that the variable ‘assets’ does have an effect on the model.  

## Linear Regression, Boostrap, and Lasso Model Comparison: 

#### Does Percentage of Canopy Cover or Age of the Forest Affect the Number of Salamanders Found?

This dataset is from: Ramsey, F.L. and Schafer, D.W. (2002). The Statistical Sleuth: A Course in Methods of Data Analysis (2nd ed), Duxbury. 

##### Description of the dataset, from the documentation:
The Del Norte Salamander (plethodon elongates) is a small (5–7 cm) salamander found among rock rubble, rock outcrops and moss-covered talus in a narrow range of northwest California. To study the habitat characteristics of the species and particularly the tendency of these salamanders to reside in dwindling old-growth forests, researchers selected 47 sites from plausible salamander habitat in national forest and parkland. Randomly chosen grid points were searched for the presence of a site with suitable rocky habitat. At each suitable site, a 7 metre by 7 metre search are was examined for the number of salamanders it contained. This data frame contains the counts of salamanders at the sites, along with the percentage of forest canopy and age of the forest in years. 

##### Variables:
* Site: Investigated site
* Salaman: Number of salamanders found in 49 square meter area
* PctCover: Percentage of Canopy Cover
* Forestage: Forest age


```R
install.packages('Sleuth2')
library(Sleuth2)
# help(case2202) # Uncomment if you want to look at the documentation for this data
```

    
    The downloaded binary packages are in
    	/var/folders/14/0286vgm17ynbvnkzv81_5hvh0000gn/T//RtmpIsfd4l/downloaded_packages


**First** I will run a linear regression of Salaman on PctCover and Forestage.


```R
m1 <- lm(Salaman ~ PctCover + Forestage, case2202)
summary(m1)
```


    
    Call:
    lm(formula = Salaman ~ PctCover + Forestage, data = case2202)
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -3.9357 -1.9303 -0.2844  0.7568  9.2906 
    
    Coefficients:
                  Estimate Std. Error t value Pr(>|t|)   
    (Intercept) -0.2847570  0.8532482  -0.334  0.74017   
    PctCover     0.0456223  0.0158100   2.886  0.00603 **
    Forestage    0.0003679  0.0029230   0.126  0.90042   
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 2.993 on 44 degrees of freedom
    Multiple R-squared:  0.2472,	Adjusted R-squared:  0.213 
    F-statistic: 7.225 on 2 and 44 DF,  p-value: 0.001935



It appears that PctCover is slightly significant, but not to the p <.05 level. 


```R
bootstrap_lm(m1)
```


<dl class=dl-horizontal>
	<dt>(Intercept)</dt>
		<dd>0.293848405799467</dd>
	<dt>PctCover</dt>
		<dd>0.0152904254798619</dd>
	<dt>Forestage</dt>
		<dd>0.00344673436653409</dd>
</dl>



This function outputs the p-value of the variables. The bootstrapped model indicates that both PctCover and Forestage are significant at p<.05. 


```R
x <- with(case2202, cbind(PctCover, Forestage))
y <- case2202$Salaman
set.seed(1)
lasso_m <- cv.glmnet(x,y)
coefficients(lasso_m, s='lambda.min')
```


    3 x 1 sparse Matrix of class "dgCMatrix"
                         1
    (Intercept) 0.17587896
    PctCover    0.03886497
    Forestage   .         


##### Conclusion

The Lasso model indicates that the PctCover variable should be included, but not the Forestage variable. The bootstrap model indicates that both variables could be siginficant. The linear regression shows PctCover to be slightly significant. In this case, I conclude that the best model is one that contains only PctCover as a variable.  


```R
m1 <- lm(Salaman ~ PctCover, case2202)
summary(m1)
```


    
    Call:
    lm(formula = Salaman ~ PctCover, data = case2202)
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -3.9688 -1.9220 -0.2974  0.7571  9.3124 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept) -0.29606    0.83918  -0.353 0.725890    
    PctCover     0.04687    0.01220   3.841 0.000381 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 2.96 on 45 degrees of freedom
    Multiple R-squared:  0.2469,	Adjusted R-squared:  0.2302 
    F-statistic: 14.76 on 1 and 45 DF,  p-value: 0.0003806


