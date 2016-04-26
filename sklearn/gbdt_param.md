---
layout: page
title: sklearn中的调参：gbt(gbdt/gbrt)
tagline: 介绍
---
{% include JB/setup %}

# 前言

大家都知道，我们在小学初中阶段学过一种工具，叫螺旋测微器。我们在测一个物品的距离时，可以按0后面的小数位依次调参数。

机器学习中的调参也类似。

# 如何调参？

参数说明，详见[api](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)。那么在具体的实例中，我们最关心的，应该是如何去进行调参呢？在参考2中，提到了一个示例，展示了如何去处理真实情况下的问题：

[数据集](http://www.analyticsvidhya.com/wp-content/uploads/2016/02/Dataset.rar)

处理step如下：

第一步：特征加工

- 1.drop掉City列，原因：太多类别
- 2.将DOB转为Age；drop掉DOB
- 3.创建EMI_Loan_Submitted_Missing，如果EMI_Loan_Submitted缺失，为1，否则为0；dop掉EMI_Loan_Submitted
- 4.drop掉EmployerName，原因：太多类别
- 5.Existing_EMI使用中值0进行计算，原因：111个值缺失
- 6.创建Interest_Rate_Missing变量，如果Interest_Rate缺失为1，否则为0；drop掉Interest_Rate
- 7.drop掉Lead_Creation_Date，原因：对结果影响小
- 8.Loan_Amount_Applied使用均值
- 9.Loan_Amount_Submitted_Missing...
- 10.Loan_Tenure_Submitted_Missing...
- 11.drop掉LoggedIn, Salary_Account
- 12.Processing_Fee_Missing。。。
- 13.Source：保留top2
- 14.执行Numerical和One-Hot-Coding编码

第二步：创建baseline model

使用缺省参数的GradientBoostingClassifier(random_state=10)
作为baseline。得到baseline的AUC。

建一个交叉验证的函数：

{% highlight python %}

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


{% endhighlight %}

建一个baseline模型，通常，一个好的baseline可以用缺省参数的GBM模型来构建。

{% highlight python %}

predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, train, predictors)

{% endhighlight %}

结果：

<figure>
	<a href="http://i1.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/1.-modelfit1.png"><img src="http://i1.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/1.-modelfit1.png" alt=""></a>
</figure>

第三步：参数调优

两种类型的参数：基于tree的参数、基于boosting的参数。

- tree参数：min_samples_split、min_samples_leaf、min_weight_fraction_leaf、max_depth、max_leaf_nodes、max_features
- boosting参数：learning_rate、n_estimators、subsample
- 杂项参数：loss、init、random_state、verbose、warm_start、presort

随着树的增加，GBM不会overfit，但如果learning_rate的值较大，会overfiting。如果减小learning_rate、并增加树的个数，在个人电脑上计算开销会很大。

记住：

- 1.选择一个**相对高的learning_rate**。缺省值为0.1，通常在0.05到0.2之间都应有效
- 2.**根据这个learning_rate，去优化树的数目**。这个范围在[40,70]之间。记住，选择可以让你的电脑计算相对较快的值。因为结合多种情况才能决定树的参数.
- 3.**调整树参数**，来决定learning_rate和树的数目。注意，我们可以选择不同的参数来定义树。
- 4.**调低learning_rate，增加estimator的数目**，来得到更健壮的模型

第3.1步 调整树参数，来确定learning_rate和n_estimaters

为了确定boosting参数，先调置一些初始参数：

- 1.min_samples_split = 500： 可以在~0.5-1%之间。考虑到倾斜类（imbalanced class），我们可以在这个范围内取个小值。
- 2.min_samples_leaf = 50：可以基于直觉进行选择。该参数用于防止overfitting，偏斜类不要使用小值。
- 3.max_depth = 8：基于target与predict的数目，选择（5-8）。有87行和49列，取8。
- 4.max_features = ‘sqrt’：根据经验法则，使用平方根.
- 5.subsample = 0.8：通常使用该值作为初始值

注意，所有上面的参数都是estimate的初始值，后面会进行调整。缺省的learning_rate为0.1，然后根据它来确定树的数目。出于该目的，我们可以使用GridSearch来确定[20,80]间的树数目，每个步长为10。

{% highlight python %}

predictors = [x for x in train.columns if x not in [target, IDcol]]
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])

{% endhighlight%}

输出结果：

{% highlight python %}

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

{% endhighlight %}

如下：

<figure>
	<a href="http://i0.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/2.-gsearch-1.png?w=1050"><img src="http://i0.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/2.-gsearch-1.png?w=1050" alt=""></a>
</figure>


可以看到，对于learning_rate=0.1时，当n_estimators=60效果最佳。60只在这里合理，但并不意味着对所有case下都合理。其它情况：

- 如果值为20上下，你可以尝试更低的learning_rate（比如：0.05），重新运行GridSearch
- 如果值过高（接近100），尝试更高的learning_rate

基于tree的参数调节

- 1.调节max_depth 和 num_samples_split
- 2.调节min_samples_leaf
- 3.调节max_features

**调节参数的次序必须注意**。应首先调节对结果有较大影响的参数。例如，首先调整max_depth 和 min_samples_split，它们对结果影响更大。

**注意**：如果在本节中使用很笨重的GridSearch，将花费15-30分钟，或者更多的时间（取决于你的系统）。

首先，我们先调节max_depth值到(5,15),step=2；然后将min_samples_split设置到(200,1000),step=200。基于直觉。当然，你也可以使用更小的范围而执行更多的迭代。

{% highlight python %}

param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

{% endhighlight %}

如下：

<figure>
	<a href="http://i2.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/3.-gsearch-2.png?w=1456"><img src="http://i2.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/3.-gsearch-2.png?w=1456" alt=""></a>
</figure>


运行完30对组合后，我们看到max_depth=9和min_samples_split=1000比较理想。注意：1000是我们的极限值。我们也可以尝试更高的值。

这里，我们取max_depth=9，而不再对min_samples_split取更高的值。这样做可能并不是最好的。这里我们仔细观察发现，max_depth=9 的值都挺好的。我们再测试min_samples_leaf，(30, 70), step=10：

{% highlight python %}

param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

{% endhighlight %}

<figure>
	<a href="http://i1.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/4.-gsearcg-3.png?w=1568"><img src="http://i1.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/4.-gsearcg-3.png?w=1568" alt=""></a>
</figure>

我们得到最优值为：min_samples_split=1200，min_samples_leaf=60. 我们看到cv score提高到0.8396了。对模型进行fit查看feature的重要性。


{% highlight python %}

modelfit(gsearch3.best_estimator_, train, predictors)

{% endhighlight %}

得到：

<figure>
	<a href="http://i1.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/5.-modelfit-2.png?w=1488"><img src="http://i1.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/5.-modelfit-2.png?w=1488" alt=""></a>
</figure>

如果你比较下该模型与baseline模型的feature重要性，就会发现我们可以从一些变量中提取值。最早的模型中，在一些变量上得取了过多的重要性，而现在这个模型则更加分散。

接着，我们调节最后的树参数：max_features: (7,19), step=2.

{% highlight python %}

param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10),
param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

{% endhighlight %}

如下：

<figure>
	<a href="http://i2.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/6.-gsearch-4.png?w=1036"><img src="http://i2.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/6.-gsearch-4.png?w=1036" alt=""></a>
</figure>

我们可以发现，最优的值为7, 刚好为feaatures数(50)的sqrt平方根。我们的初始值已经得到最佳了。当然，你可能希望尝试下更小的值，我们这里直接从7开始。最终的树参数为：

- min_samples_split: 1200
- min_samples_leaf: 60
- max_depth: 9
- max_features: 7

## 调节subsample，让你的模型有更小的learning_rate

接下去，我们再调节subsample值。采用了：0.6,0.7,0.75,0.8,0.85,0.9。

{% highlight python %}

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10,max_features=7),
param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

{% endhighlight %}

结果为：

<figure>
	<a href="http://i1.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/7.-gsearch-5.png?w=1020"><img src="http://i1.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/7.-gsearch-5.png?w=1020" alt=""></a>
</figure>

这里，我们发现0.85最佳。我们得到了所有的参数。现在，我们只需要合理调低learning_rate，并合理增加我们的estimators的数目。注意，这些树不一定时最优的解，但是一个很好的测试基准（benchmark）。

随着树的增加，执行CV以及寻找最优解的计算开销会越来越大。为了得到理想的模型效果，我们需要为设立私人的积分榜排名。这里就不做详解了。

接着，我们降低learning_rate到一半：0.05，并增加树的数目二倍到120.

{% highlight python %}

predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_1, train, predictors)

{% endhighlight %}

结果为：

<figure>
	<a href="http://i2.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/8.-tuned-1.png?w=1480"><img src="http://i2.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/8.-tuned-1.png?w=1480" alt=""></a>
</figure>


得到的LB得分为：0.844139

接着我们将learning_rate降到0.01，并将树提升到600。

{% highlight python %}

predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm_tuned_2 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_2, train, predictors)

{% endhighlight %}

得到结果为：

http://i0.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/9.-tuned-2.png

得到的LB得分为：0.848145

我们将learning_rate降到0.005，树的数目增加到1200棵。

{% highlight python %}

predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm_tuned_3 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1200,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7,
warm_start=True)
modelfit(gbm_tuned_3, train, predictors, performCV=False)

{% endhighlight %}

得到结果为：

<figure>
	<a href="http://i2.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/10.-tuned-3.png?w=1474"><img src="http://i2.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/10.-tuned-3.png?w=1474" alt=""></a>
</figure>

得分为：0.848112

我们看到，得分的降低很细微。再接着运行1500棵。

{% highlight python %}

predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm_tuned_4 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1500,max_depth=9, min_samples_split=1200, min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7,
warm_start=True)
modelfit(gbm_tuned_4, train, predictors, performCV=False)

{% endhighlight %}

得到结果为：

<figure>
	<a href="http://i0.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/11.-tuned-4.png?w=1470"><img src="http://i0.wp.com/www.analyticsvidhya.com/wp-content/uploads/2016/02/11.-tuned-4.png?w=1470" alt=""></a>
</figure>

LB得分为：0.848747

因此，我们可以很清楚地看到：从0.844提升到0.849已经是一个巨大的提升了。

GBM的另一个调参的hack是：warm_start。你可以使用它来增加每一步estimators的数目，并在运行过程中测试不同值。


参考：

1.[http://scikit-learn.org/stable/modules/ensemble.html](http://scikit-learn.org/stable/modules/ensemble.html)

2.[http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/](http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

3.[http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/](http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
