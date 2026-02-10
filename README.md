PCA on Handwritten Digits (MNIST / Sklearn Digits)
This project applies Principal Component Analysis (PCA) for dimensionality reduction on a handwritten digits dataset and evaluates the impact on Logistic Regression classification performance.

1. Project Overview
Dataset: Sklearn load_digits() (8×8 images flattened to 64 features) or MNIST (28×28 images).

Goal:

Compress high‑dimensional image features using PCA.

Analyze explained variance to choose a good number of components.

Compare Logistic Regression accuracy before and after PCA.

2. Dataset
Source:

Primary: sklearn.datasets.load_digits() built‑in dataset.

Alternative: MNIST from external source (e.g., Kaggle / keras.datasets).

Shape (digits):

Features X: (n_samples, 64)

Labels y: (n_samples,), digits 0–9.

3. Methods and Workflow
Load dataset

Load digits/MNIST and flatten images into 1D feature vectors per sample.

Train–test split

Split into train and test sets (e.g., 80/20) with train_test_split and stratified labels.
​

Feature scaling

Apply StandardScaler to standardize features (zero mean, unit variance) before PCA so all features contribute fairly.

PCA for dimensionality reduction

Fit PCA with n_components=None on the scaled training data to get all components.

Inspect explained_variance_ratio_ and its cumulative sum to see how many components capture 90–95% of the variance.

Experiment with component counts: 2, 10, 30, 50.

Explained variance plot (Deliverable 1)

Plot cumulative explained variance vs number of components.

Use this plot to justify your chosen dimensionality.

Reduced datasets (Deliverable 2)

Transform train and test sets using PCA to create reduced feature sets for each n_components value.

Optionally export reduced datasets to CSV for reuse.

Logistic Regression models

Baseline model on original scaled features.

Separate models on PCA‑reduced features with 2, 10, 30, 50 components.

Accuracy comparison (Deliverable 3)

Record accuracy for:

Original features

PCA(2), PCA(10), PCA(30), PCA(50)

Summarize results in a small table and discuss trade‑off between dimensionality and performance.

Visualization

Create a 2D scatter plot of the first two principal components, colored by digit label, to see separation between classes.

4. File Structure
Example structure:

text
PCA_Digits/
│
├── MNIST digits.ipynb          # Main notebook with all steps
├── data/
│   └── digits_train_pca_30.csv   # Example reduced dataset (optional)
│
├── outputs/
│   ├── explained_variance_plot.png
│   └── pca_2d_scatter.png

You can adjust filenames and folders as needed.

5. Key Results
Update this section with your actual numbers.

Baseline Logistic Regression (original 64 features):

Test accuracy: 0.9722222222222222

PCA + Logistic Regression results:

Features	Components	Test accuracy

PCA(2 components) accuracy: 0.5250
PCA(10 components) accuracy: 0.8778
PCA(30 components) accuracy: 0.9556
PCA(50 components) accuracy: 0.9667
Observation examples:

With around 30–50 components, model retains most of the variance and achieves accuracy close to the original feature space but with fewer dimensions.

Very low dimensions (e.g., 2 components) are useful for visualization but may reduce accuracy.

6. What I Learned
PCA can compress high‑dimensional image data into a smaller set of orthogonal components while preserving most of the important variance.

Explained variance helps decide how many components to keep by quantifying information retained at each dimensionality.

Scaling is essential before PCA; without it, features with larger magnitude dominate the principal components.
​

Logistic Regression on PCA‑reduced data can be faster and still perform competitively, showing a trade‑off between dimensionality, speed, and accuracy.

7. How to Run
Clone the repository:

git clone <https://github.com/PranithaBokketi/task13-MNIST-digits>.git
Create and activate a virtual environment (optional but recommended).

Install dependencies:

pip install -r requirements.txt
(Or manually: pip install numpy pandas scikit-learn matplotlib.)
​

Open the notebook:


jupyter notebook pca_digits.ipynb
Run all cells in order to reproduce preprocessing, PCA, plots, and model training.

8. Possible Extensions
Try different classifiers after PCA (e.g., SVM, KNN) and compare with Logistic Regression.

Use GridSearchCV with a pipeline (Scaler → PCA → Logistic Regression) to tune both n_components and regularization parameter C.

Repeat the experiment on full MNIST (28×28, 784 features) to see PCA’s impact on much higher dimensional data.
​
