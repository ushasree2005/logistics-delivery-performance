# Retail and Logistics Analysis with TensorFlow

Two TensorFlow models on two datasets, each tested on data it never saw and compared with a simple baseline. A model only counts as useful if it beats the baseline.

**Kaggle notebook:** PASTE_KAGGLE_LINK_HERE
**Datasets:** [Retail Transactions Dataset](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset) (1,000,000 baskets) · [Transportation and Logistics Tracking Dataset](https://www.kaggle.com/datasets/nicolemachado/transportation-and-logistics-tracking-dataset) (6,880 trips)

## Part A: Retail, "complete the basket"
Hide one product from each basket and ask a neural network to guess it from the other products plus store, season, city and customer type.

| Method | Hidden product in top 5 guesses |
|---|---|
| Random guess | 6.3% |
| Most popular products (baseline) | 7.6% |
| TensorFlow model | 7.5% |

**Finding:** the model can't beat the baseline because the data holds no pattern to learn. Market basket analysis confirms it: none of the 3,240 product pairs is bought together more than chance (highest lift 0.997), the product mix doesn't differ by store, season, city or customer type (all p > 0.2), and the Season label doesn't match the purchase date. The dataset is synthetic, so the honest conclusion is that it can't support "bought together" promotions.

## Part B: Logistics, "will this trip be late?"
Predict at **booking time** (customer, supplier, truck, route, cargo, distance, planned hours) whether a GPS-tracked trip will arrive late.

| Test | Customer rule | Logistic regression | **TensorFlow** |
|---|---|---|---|
| Newest 20% of trips (main) | 0.715 | 0.933 | **0.939** |
| Random 80/20 | 0.847 | 0.957 | **0.972** |
| Whole days held out | 0.854 | **0.955** | 0.944 |

ROC-AUC on held-out trips.

**Findings**
- The model ranks late trips well on **future** trips (AUC 0.939), about as well as logistic regression and far better than a simple rule.
- The biggest driver is **how tight the plan is**: plans needing more than 10 km/h door to door are late 84–96% of the time, versus 11% at 5 km/h or less.
- In practice, the top-risk band is **24% of trips but catches 82% of late ones**.

## Tools
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, SciPy, Matplotlib

## Files
- `retail_logistics_tensorflow.ipynb`: full notebook (basket analysis, both TensorFlow models, baselines, charts)
