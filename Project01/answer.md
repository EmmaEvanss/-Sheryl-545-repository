#### problem 1

A
Mean: 0.05019795790476916
Variance: 0.010332476407479581
Skewness: 0.1204447119194402
Kurtosis: 0.2229270674503816
B

**Normal distribution.**

Based on the data's skewness (0.12) and kurtosis (0.22), the distribution is nearly symmetric with light tails, closely resembling a normal distribution. Thus, a normal distribution is a better choice for modeling this dataset as it aligns well with the observed characteristics.

C

Both the normal and t-distributions were fitted to the data. The normal distribution closely matches the histogram with mean 0.05 and standard deviation 0.10. The t-distribution, with 28.71 degrees of freedom, also fits well but offers no significant advantage as the data lacks heavy tails. **This supports the earlier choice of the normal distribution.**

![image-20250204203956108](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250204203956108.png)

#### problem2

A

![image-20250204204544548](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250204204544548.png)

B

No, the covariance matrix is not at least positive semi-definite because it has negative eigenvalues.

A matrix is positive semi-definite if all its eigenvalues are non-negative.

C

The two matrices using different methods share the same results.

![image-20250204211539475](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250204211539475.png)



![image-20250204205401816](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250204205401816.png)

D

![image-20250204205618637](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250204205618637.png)

E

Matrix in question C ensures mathematical validity through adjusting negative eigenvalues, so it has higher variances and covariances.

Matrix in question D is calculated based on overlapping data from the original dataset, reflecting true data relationships but with smaller values. So it has lower variances and covariances.

#### problem3

A

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250204213518779.png" alt="image-20250204213518779" style="zoom:50%;" />



![image-20250204214300608](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250204214300608.png)

B

![image-20250206203917728](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250206203917728.png)

C

The simulation checks the conditional distribution X2∣X1=0.6 by calculating its theoretical mean and variance using the properties of the multivariate normal distribution, verifying the results through sampling. A large number of samples are generated from the joint distribution using the covariance matrix and mean vector, and X2values corresponding to X1 near 0.6 are filtered. Additionally, conditional samples of X2 are directly generated from the conditional distribution formula, and their mean and variance are compared with the theoretical values to validate consistency.

![image-20250206230313068](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250206230313068.png)

The simulated mean and variance are nearly the same as the results in question B.

#### problem4

A

![image-20250204221240225](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250204221240225.png)

**MA(1)**:

- **ACF**: Significant at lag 1, then quickly diminishes.
- **PACF**: Only significant at lag 1.

**MA(2)**:

- **ACF**: Significant at lags 1 and 2, then diminishes.
- **PACF**: Significant at lags 1 and 2, others are insignificant.

**MA(3)**:

- **ACF**: Significant at lags 1, 2, and 3, then diminishes.
- **PACF**: Significant at lags 1, 2, and 3, others are insignificant.

B

![image-20250205220347947](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250205220347947.png)

**AR(1)**:

- **ACF**: Exponential decay.
- **PACF**: Significant at lag 1 only.

**AR(2)**:

- **ACF**: Exponential decay.
- **PACF**: Significant at lags 1 and 2.

**AR(3)**:

- **ACF**: Exponential decay.
- **PACF**: Significant at lags 1, 2, and 3.



C

The AR(3) model is more appropriate for modeling the data.

For the MA model, both ACF and PACF show a cutoff, which does not align with the theoretical behavior of an MA process, where ACF should cut off while PACF decays. Therefore, the MA model is likely not a good fit for the data.

However with the AR model,PACF shows a clear cutoff, while ACF exhibits a gradual decay, which matches the expected pattern of an AR process.

AR(3) is the best among others because PACF cuts off at lag 3, ACF exhibits a relatively slow exponential decay. Both aligns with the theoretical characteristics.

D

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250206010903600.png" alt="image-20250206010903600" style="zoom:50%;" />

The lower the AICc, the better the model. 

AR(3) model has the lowest AICc value(-1746.26), making it the best fit.



#### problem5

A

I created a routine for calculating an exponentially weighted covariance matrix by applying exponential decay weights to the centered data and summing the weighted products for each pair of columns. The results from my routine were compared to those from pandas.ewm.cov. Both approaches produced consistent structures and similar numerical values for the covariance matrix. Although minor discrepancies were observed, they likely stem from differences in the specific handling of weights and normalization. Based on this comparison, I conclude that my routine is accurate and produces the expected results.

![image-20250206231416610](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250206231416610.png)

![image-20250206231429906](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250206231429906.png)



B

![image-20250207191817995](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250207191817995.png)

C

The values of λ have a significant impact on the covariance matrix by determining how past data points are weighted. When λ is close to 1, the covariance matrix heavily prioritizes recent data, making it more reflective of short-term dynamics. This results in a covariance structure where fewer eigenvalues explain most of the variance, as seen in the steep cumulative variance curves for high λ. Conversely, when λ is lower, the weighting is more evenly distributed across all data points, leading to a more balanced covariance matrix that requires more components to explain the same level of variance. This behavior highlights how λ can be used to adjust the sensitivity of the covariance matrix to recent versus historical data.



#### problem6

A

Cholesky Simulation Shape: (10000, 500)

B

PCA Simulation Shape: (10000, 500)

C

Frobenius Norm (Cholesky): 0.021175

Frobenius Norm (PCA): 0.083185

The Frobenius norm for the Cholesky method is 0.021175, indicating that the simulated covariance matrix closely approximates the original one. This result is expected because the Cholesky decomposition retains the full structure of the covariance matrix, ensuring high accuracy in the generated samples. 

In contrast, the PCA method, which retained 29 principal components to explain 75% of the variance, resulted in a significantly larger Frobenius norm of 0.083185. This suggests that the PCA simulation introduces a notable loss of information due to dimensionality reduction, leading to a covariance matrix that deviates more from the original.

D

![image-20250207005532316](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250207005532316.png)

The Cholesky simulated covariance aligns almost perfectly with the original, indicating that the Cholesky method retains the entire covariance structure. On the other hand, the PCA simulated covariance curve rapidly reaches the 75% variance threshold, using only a fraction of the components. This efficiency in dimensionality reduction highlights PCA's ability to capture dominant variance patterns but also reveals that it sacrifices finer details of the covariance structure, as seen by the sharp leveling-off after the threshold is reached. Overall, PCA is faster and focuses on key features, while Cholesky offers a faithful reconstruction at the cost of computational intensity.

E

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20250208011031569.png" alt="image-20250208011031569" style="zoom:50%;" />

By reducing the number of components and focusing only on the most significant sources of variance (29 components for PCA vs. 500 for Cholesky), PCA significantly decreases the time required to simulate the data. The Cholesky method, on the other hand, retains the full covariance structure, leading to a longer runtime due to the higher dimensionality.

F

PCA is faster because it reduces dimensionality by focusing only on the components that explain the majority of the variance, making it computationally efficient. In contrast, Cholesky retains the full covariance structure, preserving all details but taking more time. While PCA is ideal for tasks where speed and the most significant variance patterns are sufficient, it sacrifices some of the finer details in the covariance matrix. Cholesky, however, is better suited for scenarios where full accuracy and preservation of the covariance structure are critical, despite the higher computational cost. The tradeoff between speed and fidelity depends on the specific requirements of the task at hand.







