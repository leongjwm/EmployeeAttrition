# EmployeeAttrition
A machine learning project done by my groupmates and I in `R` Programming Language, where we attempt to predict whether an employee would attrite or otherwise.

My contributions in this project were training various decision tree ensembles, i.e. Random Forests, Gradient Boosted Decision Trees and Bagged Trees.

`EmployeeAttrition_4248_NOSMOTE.R` means that the models were trained without using [SMOTE](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis). As expected, due to the imbalanced dataset, the sensitivity of the class `1` (the employees that attrited) is low. 

`EmployeeAttrition_4248_SMOTE.R` means that the models were trained using SMOTE, which showed much more promising results.

Metrics used were accuracy, sensitivity, specificity and AUC score (area under curve).
