Loan Eligibility & Max EMI Predictor
Enter the applicant's financial details below to get a risk assessment and a maximum recommended EMI amount.

Applicant Financial Data
Age (Years)

35


Monthly Salary ($)

10000.00


Years of Employment

10.00


Monthly Rent/Mortgage ($)

1500.00


School Fees ($)

0.00


College Fees ($)

0.00


Travel Expenses ($)

200.00


Groceries/Utilities ($)

800.00


Other Expenses ($)

300.00


Current EMI Amount ($)

500.00


Credit Score (FICO)

750


Bank Balance ($)

25000.00


Emergency Fund ($)

10000.00


Loan Requested Amount ($)

50000.00


Max Monthly EMI (Placeholder value)

1500.00


Calculated Total Monthly Expenses

$3,300.00
Calculated EMI/Salary Ratio

0.05

ValueError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/emi_predictor/app.py", line 218, in <module>
    main()
    ~~~~^^
File "/mount/src/emi_predictor/app.py", line 180, in main
    eligibility_prediction = predict_eligibility(scaled_data_for_model, classifier_model, label_encoder)
File "/mount/src/emi_predictor/app.py", line 80, in predict_eligibility
    prediction_encoded = model.predict(scaled_features)
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py", line 774, in inner_f
    return func(**kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/sklearn.py", line 1839, in predict
    class_probs = super().predict(
        X=X,
    ...<3 lines>...
        iteration_range=iteration_range,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py", line 774, in inner_f
    return func(**kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/sklearn.py", line 1443, in predict
    predts = self.get_booster().inplace_predict(
        data=X,
    ...<4 lines>...
        validate_features=validate_features,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py", line 774, in inner_f
    return func(**kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/xgboost/core.py", line 2865, in inplace_predict
    raise ValueError(
    ...<2 lines>...
    )s
