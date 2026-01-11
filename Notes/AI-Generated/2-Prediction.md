# Prediction Problems in Business Research

**Course:** DOTE 6635: Artificial Intelligence for Business Research (Spring 2026)
**Instructor:** Renyu (Philip) Zhang

## Introduction: The Ubiquity and Utility of Prediction

Prediction is a fundamental component of decision-making in a vast array of contexts, from macroeconomic policy to individual consumer choices. The ability to accurately forecast future outcomes is of paramount importance to researchers, policymakers, and business leaders alike. This document, based on the lecture slides from "Prediction Problems in Business Research," explores the critical role of prediction, its applications across various domains, and the theoretical underpinnings that distinguish it from related concepts like estimation and causal inference.

## Why We Care About Predictions

The importance of prediction can be understood through three primary lenses. First, we are intrinsically interested in forecasting **macro-level outcomes** that affect society at large. These include, but are not limited to, population growth, election results, Gross Domestic Product (GDP), poverty rates, and the impact of tax policies. Second, in many instances, **good predictions directly inform good decisions and policies**. For example, accurate weather forecasts can save lives and property, while precise demand forecasting can optimize supply chains and minimize waste. In the financial sector, predicting stock and asset returns is the bedrock of investment strategies. Recommendation systems, which predict user preferences, are now a ubiquitous feature of e-commerce and content platforms. In healthcare, predicting patient lifetime value (LTV) or the likelihood of disease can lead to better resource allocation and preventative care.

Third, prediction is a cornerstone of **causal inference**. To estimate the causal effect of an intervention, we must predict the counterfactual outcome—what would have happened in the absence of the intervention. This is the foundation of many modern causal machine learning techniques, such as Double Machine Learning (DML), honest trees, and matrix completion methods.

A key insight from Kleinberg et al. (2015) is the decomposition of a policy problem into prediction and causation components. The change in a policy outcome (π) with respect to a policy lever (X₀) can be expressed as:

```
dπ(X₀, Y)/dX₀ = ∂π/∂X₀ (Y) [prediction] + ∂π/∂Y · ∂Y/∂X₀ [causation]
```

This equation highlights that the overall policy effect is a sum of a direct effect (the prediction component) and an indirect effect mediated by the outcome variable Y (the causation component) [1].

## Applications of Prediction in Research

The lecture slides showcase a variety of research papers that leverage prediction for novel insights.

### Macro Predictions

In the realm of macroeconomics, Jean et al. (2016) demonstrate the power of combining satellite imagery with machine learning to predict poverty. By using nighttime light intensity as a proxy for economic activity, their convolutional neural network (CNN) model can explain up to 75% of the variation in local-level economic outcomes in developing countries [2]. Similarly, van Binsbergen et al. (2023) constructed a 170-year time series of economic sentiment by applying machine learning to a massive corpus of 200 million pages from U.S. local newspapers. This sentiment index proved to be a powerful predictor of macroeconomic fundamentals like GDP, consumption, and employment growth [3].

### Demand Forecasting

Demand forecasting has been revolutionized by the availability of more and wider data, as well as the shift from linear models to more sophisticated machine learning techniques. Bajari et al. (2015) explored the use of machine learning methods for demand estimation, showing their superiority over traditional linear models [4]. Cui et al. (2018) found that incorporating social media information can improve demand forecast accuracy by a significant margin of 12.65% to 23.23% [5]. Cohen et al. (2022) introduced a novel Data Aggregation with Clustering (DAC) method to improve demand prediction by effectively aggregating data from various sources [6].

### Recommendation Systems

Recommendation systems are a prime example of prediction in a business context. Farias and Li (2019) developed a tensor recovery approach that incorporates side information to learn user preferences more effectively, leading to better personalization [7]. Peng and Liang investigated the differences between view-based (view-also-view) and purchase-based (purchase-also-purchase) recommender systems. Their findings indicate that while view-based systems are more effective at generating views and sales for a wider range of products, purchase-based systems are more effective for cheaper products [8].

From a computer science perspective, Zhan et al. (2022) addressed the issue of duration bias in watch-time prediction for video recommendations on platforms like Kuaishou. They proposed a Duration-Deconfounded Quantile-based (D2Q) framework to mitigate this bias and improve recommendation quality [9]. Covington, Adams, and Sargin provided insights into the architecture of YouTube's massive recommendation system, which employs a two-stage process of candidate generation and deep ranking [10].

### Other Prediction Domains

The application of prediction extends to numerous other fields. In finance, Gu, Kelly, and Xiu have shown that machine learning models, particularly trees and neural networks, can double the performance of traditional regression-based strategies for empirical asset pricing [11]. In medicine, a deep learning system called PANDA has demonstrated remarkable accuracy in detecting pancreatic cancer from non-contrast CT scans, achieving an AUC of 0.986-0.996 and outperforming human radiologists [12]. The development of AlphaFold 3 represents a monumental leap in predicting the three-dimensional structure of proteins, which has profound implications for drug discovery and biology [13].

## From Prediction to Decision

A crucial application of prediction is in informing high-stakes decisions. Kleinberg et al. (2018) studied the use of machine predictions in judicial bail decisions. Their research showed that using a machine learning model to predict a defendant's flight risk could lead to a reduction in crime rates of up to 24.7% without changing the jailing rate, or a reduction in the jailing rate of up to 41.9% without an increase in crime [14]. This highlights the potential for algorithmic decision support to improve outcomes in complex social systems.

## When Do Predictions Make No Sense?

Despite its power, prediction is not a panacea. There are several scenarios where prediction models may be ineffective or even misleading.

1.  **Lack of Importance:** The prediction target is not a sufficiently important macroeconomic, political, or natural outcome.
2.  **Inaccuracy or Lack of Causality:** The prediction is neither accurate enough nor causally relevant for decision-making. As the previously mentioned formula shows, if the prediction of Y is inaccurate, or if the causal link between the policy and the outcome is not well-identified, the resulting policy decision will be flawed.
3.  **Ungrounded Counterfactuals:** The predictions of counterfactual outcomes are unreliable due to the violation of key assumptions for causal inference:
    *   **Unconfoundedness (Conditional Independence Assumption - CIA):** The treatment assignment is independent of the potential outcomes, conditional on a set of observed covariates.
    *   **Common Support (Overlapping Condition):** For any given set of covariates, there is a non-zero probability of being both treated and untreated.

## Prediction vs. Estimation: A Broader Perspective

It is essential to distinguish prediction from the related concept of estimation. Hofman et al. (2021) provide a useful framework for thinking about the interplay between explanation (causal inference) and prediction in computational social science [15]. They propose a 2x2 matrix that categorizes different research goals based on whether they involve an intervention and whether the focus is on specific features or outcome prediction.

|                          | Without Intervention                              | With Intervention                                      |
|--------------------------|------------------------------------------------|------------------------------------------------------|
| Specific Features/Effects| Descriptive analysis or constructing new measurements | Causal inference or applied micro                    |
| Outcome Prediction       | Predictive modeling or forecasting             | Structural estimation, counterfactual simulation and world model |

*Table 1: A Framework for Integrating Explanation and Prediction in Computational Social Science (Adapted from Hofman et al., 2021)*

This framework clarifies that **predictive modeling and forecasting** are concerned with predicting outcomes *without* an intervention. In contrast, when an intervention is involved, the goal shifts to **structural estimation, counterfactual simulation, and building world models** to understand the causal impact of the intervention on the outcome.

## Conclusion

Prediction is a powerful and versatile tool that is transforming business research and practice. From forecasting macroeconomic trends to personalizing user experiences and informing critical policy decisions, the applications of prediction are vast and growing. However, it is crucial to understand the limitations of prediction and the assumptions that underpin its use, particularly when it is used to inform causal inferences and policy interventions. By integrating prediction with explanation, as proposed by Hofman et al. (2021), researchers can build more robust and reliable models of the world, leading to better decisions and a deeper understanding of complex social and economic phenomena.

## References

[1] Kleinberg, J., Ludwig, J., Mullainathan, S., & Obermeyer, Z. (2015). Prediction Policy Problems. *American Economic Review*, 105(5), 491-495. https://doi.org/10.1257/aer.p20151023
[2] Jean, N., Burke, M., Xie, M., Davis, W. M., Lobell, D. B., & Ermon, S. (2016). Combining satellite imagery and machine learning to predict poverty. *Science*, 353(6301), 790-794. https://doi.org/10.1126/science.aaf7894
[3] van Binsbergen, J. H., Bryzgalova, S., Mukhopadhyay, M., & Sharma, V. (2023). (Almost) 200 Years of News-Based Economic Sentiment. *SSRN*. http://dx.doi.org/10.2139/ssrn.4398326
[4] Bajari, P., Nekipelov, D., Ryan, S. P., & Yang, M. (2015). Machine Learning Methods for Demand Estimation. *American Economic Review*, 105(5), 481-485. https://doi.org/10.1257/aer.p20151021
[5] Cui, R., Gallino, S., Moreno, A., & Zhang, D. J. (2018). The Operational Value of Social Media Information. *Production and Operations Management*, 27(7), 1749-1769. https://doi.org/10.1111/poms.12800
[6] Cohen, M. C., Zhang, R., & Jiao, K. (2022). Data Aggregation and Demand Prediction. *Operations Research*, 70(5), 2635-2657. https://doi.org/10.1287/opre.2022.2301
[7] Farias, V. F., & Li, A. A. (2019). Learning Preferences with Side Information. *Management Science*, 65(8), 3731-3749. https://doi.org/10.1287/mnsc.2018.3092
[8] Peng, J., & Liang, C. (2021). On the Differences Between View-Based and Purchase-Based Recommender Systems. *MIS Quarterly*, 45(2), 939-956.
[9] Zhan, R., Pei, C., Su, Q., et al. (2022). Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation. *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 2472-2481. https://doi.org/10.1145/3534678.3539092
[10] Covington, P., Adams, J., & Sargin, E. (2016). Deep Neural Networks for YouTube Recommendations. *Proceedings of the 10th ACM Conference on Recommender Systems*, 191-198. https://doi.org/10.1145/2959100.2959190
[11] Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223-2273. https://doi.org/10.1093/rfs/hhz136
[12] An, C., et al. (2023). Large-scale pancreatic cancer detection via non-contrast CT and deep learning. *Nature Medicine*, 29, 1206-1215. https://doi.org/10.1038/s41591-023-02294-y
[13] DeepMind. (2024). *AlphaFold 3*. https://www.deepmind.com/alphafold
[14] Kleinberg, J., Lakkaraju, H., Leskovec, J., Ludwig, J., & Mullainathan, S. (2018). Human Decisions and Machine Predictions. *The Quarterly Journal of Economics*, 133(1), 237-293. https://doi.org/10.1093/qje/qjx032
[15] Hofman, J. M., Watts, D. J., Athey, S., Garip, F., Griffiths, T. L., Kleinberg, J., ... & Yarkoni, T. (2021). Integrating explanation and prediction in computational social science. *Nature*, 595(7866), 1-9. https://doi.org/10.1038/s41586-021-03659-0
