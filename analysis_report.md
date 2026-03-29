# Analysis Report: Data Preprocessing and Linear Transformations

## 1) Interpretation of Main Results
Scaling changes model performance by altering optimization geometry and feature balance.
Across datasets, the best configuration depends on data distribution and model type.

Top configurations observed:
- Dataset=Breast Cancer, Model=KNN: MinMaxScaler (accuracy=0.9701 +/- 0.0070)
- Dataset=Breast Cancer, Model=LogisticRegression: StandardScaler (accuracy=0.9789 +/- 0.0070)
- Dataset=Synthetic Classification, Model=KNN: MinMaxScaler (accuracy=0.9283 +/- 0.0174)
- Dataset=Synthetic Classification, Model=LogisticRegression: StandardScaler (accuracy=0.8542 +/- 0.0253)
- Dataset=Synthetic Text (TF-IDF sparse), Model=LogisticRegression: No scaling (accuracy=1.0000 +/- 0.0000)

## 2) Outlier Robustness
RobustScaler should degrade less than StandardScaler when heavy outliers are injected.
Observed outlier experiment:
- Clean | StandardScaler: accuracy=0.9789 +/- 0.0070
- Clean | RobustScaler: accuracy=0.9789 +/- 0.0070
- With Outliers | StandardScaler: accuracy=0.9156 +/- 0.0162
- With Outliers | RobustScaler: accuracy=0.9297 +/- 0.0147

## 3) Sparse Data Discussion
TF-IDF features are sparse. MaxAbsScaler scales magnitudes while preserving zeros.
StandardScaler(with_mean=True) is not suitable for sparse data because centering destroys sparsity and is memory-inefficient.
Observed StandardScaler sparse error: 
All the 5 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
5 fits failed with the following error:
Traceback (most recent call last):
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\model_selection\_validation.py", line 833, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\base.py", line 1336, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\pipeline.py", line 613, in fit
    Xt = self._fit(X, y, routed_params, raw_params=params)
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\pipeline.py", line 547, in _fit
    X, fitted_transformer = fit_transform_one_cached(
                            ~~~~~~~~~~~~~~~~~~~~~~~~^
        cloned_transformer,
        ^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        params=step_params,
        ^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\joblib\memory.py", line 326, in __call__
    return self.func(*args, **kwargs)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\pipeline.py", line 1484, in _fit_transform_one
    res = transformer.fit_transform(X, y, **params.get("fit_transform", {}))
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\utils\_set_output.py", line 316, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\base.py", line 910, in fit_transform
    return self.fit(X, y, **fit_params).transform(X)
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\preprocessing\_data.py", line 924, in fit
    return self.partial_fit(X, y, sample_weight)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\base.py", line 1336, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "C:\Users\Kholoud\Documents\4IIR\Exploration De Données\mini_project_scaling\.venv\Lib\site-packages\sklearn\preprocessing\_data.py", line 990, in partial_fit
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.


## 4) Pipelines and Generalization
Pipelines prevent data leakage by fitting transformations only inside each CV train fold.
This improves trustworthiness of estimated generalization performance.

## 5) Bonus: PCA and Coefficient Stability
PCA can reduce dimensionality/noise but may trade slight accuracy for speed and regularization effects.
Coefficient stability indicates how sensitive learned parameters are to resampling.
Lower mean coefficient standard deviation suggests more stable estimates.

PCA summary:
- StandardScaler + LogisticRegression: accuracy=0.9789 +/- 0.0070, time=0.013s
- StandardScaler + PCA(95%) + LogisticRegression: accuracy=0.9807 +/- 0.0066, time=0.024s

Coefficient stability summary:
- No scaling: coefficient_std_mean=0.070919
- RobustScaler: coefficient_std_mean=0.121627
- StandardScaler: coefficient_std_mean=0.131561

## 6) Engineering Conclusions
- Use scaling by default for linear and distance-based models.
- Prefer RobustScaler in outlier-heavy settings.
- Use MaxAbsScaler for sparse text features.
- Keep preprocessing inside sklearn Pipeline to avoid leakage and improve reproducibility.