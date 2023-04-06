# Causal inference demo with Gurobi Machine Learning

This repository demonstrates the usage of [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/).

Gurobi Machine Learning is a library that integrates mathematical optimization and machine learning. This project explores its usage in the context of causal inference.

## Background

The integration of mathematical optimization and machine learning has gained increasing attention due to their complementary nature in solving complex problems.

Mathematical optimization involves the use of mathematical techniques to determine the best solution for a given problem, while machine learning involves the use of algorithms to learn patterns from data and make predictions or decisions. There are several ways in which mathematical optimization and machine learning can be combined.

One approach is to use trained machine learning models as variables in mathematical optimization. [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/), which was released in November 2022, is an example of this approach. This method can speed up the calculation of mathematical optimization when dealing with large datasets.

![ML2MO](https://user-images.githubusercontent.com/2261067/230796668-c48b1c84-58c0-40b8-b671-8e08aa5306f5.png)

Another approach is to use mathematical optimization as inputs to machine learning. This is exemplified by [MIPLearn](https://anl-ceeesa.github.io/MIPLearn/), which combines mixed-integer programming (MIP) with machine learning algorithms. This approach can help to improve the performance of machine learning models by incorporating optimization techniques.

![MO2ML](https://user-images.githubusercontent.com/2261067/230796778-960829f6-1c9f-4c98-96b6-bee1c5cd47f4.png)

Machine learning can also be used to optimize tuning parameters in mathematical optimization, which is another way to integrate these fields. Additionally, mathematical optimization can be used to select features and choose the best machine learning model.

However, when machine learning models is used in the objective function or constraints of mathematical optimization, treating the machine learning model as a black box can be too slow to compute. Therefore, alternative methods need to be explored to integrate these fields more effectively.

In this project, we explore [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/) for using machine learning models in mathematical optimization.

## Gurobi Machine Learning

[Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/) formulates machine learning models in mixed-integer programming (MIP).

This library supports the use of machine learning models in mathematical optimization by translating them into mathematical expressions. However, some models may require approximation to fit into the MIP framework. For example, logistic regression cannot be represented directly in Gurobi using linear constraints, so piecewise-linear approximation is used instead.

There are limitations to the machine learning models that can be used in [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/). Currently, it only supports certain models from [scikit-learn](https://scikit-learn.org/stable/), such as [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression), [PLSRegression](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html#sklearn.cross_decomposition.PLSRegression), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression), [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor), [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor), [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor), and [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor). [Keras](https://keras.io/) and [PyTorch](https://pytorch.org/) are also supported, but only Linear and Relu layers for activation functions are supported.

Unsupported models, such as LightGBM, will generate following error message.

<img width="646" alt="Screenshot 2023-04-09 at 17 54 36" src="https://user-images.githubusercontent.com/2261067/230795104-01e688fb-4057-491e-bcd7-26d438144c29.png">

## Causal Inference

The application of [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/) is particularly advantageous in the context of causal inference, as it allows for the incorporation of causal relationships between variables into the optimization process. Causal inference is a fundamental task in many fields, such as economics, social sciences, and healthcare, where understanding the effects of interventions is crucial.

One popular approach to causal inference in machine learning is uplift modeling, which is a technique used to identify the difference in treatment effect between two groups.

![Uplift](https://user-images.githubusercontent.com/2261067/230800084-1d64bfd7-18d6-4dc5-a3ea-8df304d02700.png)

It helps determine whether a treatment will have an impact on each individual instance. However, uplift modeling may not always provide sufficient guidance for decision-making.

In contrast, the integration of mathematical optimization and machine learning with [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/) can help answer more complex causal inference questions that involve decision-making under budget and other constraints. For example, in the [student enrollment model provided by Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/mlm-examples/student_admission.html), the goal is to determine which scholarships to offer to maximize enrollment while staying within budget limits.

In this way, [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/) can offer a more comprehensive solution to the problem of causal inference.

## Demo

[1. Causal inference demo with Gurobi Machine Learning](https://github.com/yujiosaka/causal-inference-demo-with-gurobi-machine-learning/blob/main/lab/1.%20Causal%20inference%20demo%20with%20Gurobi%20Machine%20Learning.ipynb)

## Conclusion

The integration of mathematical optimization and machine learning has a lot of potential for causal inference in specific situations.

However, there are limitations such as the fact that only [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) is currently supported as a binary classifier in [scikit-learn](https://scikit-learn.org/stable/), which may not be sufficient for solving complex real-world problems.

During the investigation, it was discovered that [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/) supports each regressor and classifier separately instead of treating everything as black boxes. This means that Gurobi must support each model individually, which can be time-consuming.

It is worth noting that this feature is a recent addition to Gurobi, introduced in version 10.0, released in November 2022. Therefore, it is essential to stay updated in this area and keep an eye out for any future developments.
