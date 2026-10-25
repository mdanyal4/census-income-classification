# Census Income Classification

A Python notebook comparing three machine learning models for predicting whether annual income exceeds $50,000 from demographic and employment data.

## Approach

1. Replace missing-value markers (`?`) and remove incomplete rows.
2. Encode income labels and one-hot encode categorical features.
3. Split the data into 80% training and 20% testing sets.
4. Train a decision tree, a 5-neighbor KNN classifier, and a random forest.
5. Compare test accuracy and plot the random forest's five highest-ranked features.

For KNN, the scaler is fitted on the training set and then applied to both sets.

## Saved notebook results

| Model | Test accuracy |
| --- | ---: |
| Decision tree | 80.36% |
| K-nearest neighbors | 80.74% |
| Random forest | 84.00% |

These are the outputs saved in the committed notebook, using 24,129 training rows and 6,033 test rows. The split and tree-based models do not specify random seeds, so rerunning can produce different results.

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, and Jupyter / Google Colab.

## Run locally

1. Download or clone this repository.
2. Place `adult.csv` alongside the notebook. The dataset is not included. It needs column headers and an `income` column containing `>50K` and `<=50K` labels; a raw headerless dataset requires preparation first.
3. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook income_classification.ipynb
   ```

5. Run the notebook from top to bottom.

Alternatively, upload [income_classification.ipynb](income_classification.ipynb) and the CSV to Google Colab.

## Outputs and scope

The notebook generates income-distribution, feature-importance, and model-comparison charts. This is an exploratory model comparison, not a deployed service. Evaluation uses a single held-out split; cross-validation, class-specific metrics, and fairness analysis are not implemented. Feature importance describes the fitted model rather than causal effects on income.
