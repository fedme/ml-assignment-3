# ml-assignment-3

## Install packages
1. Install latest version of Anaconda 3 from https://www.anaconda.com/
2. Clone the project repository from https://github.com/fedme/ml-assignment-3
3. Open an Anaconda 3 prompt
4. From the prompt, browse to the folder where you cloned the repository
5. Run the following command to create the conda environment:
    ```
    conda env create --file=environment.yaml
    ```
6. Activate the newly created environment:
    ```
    activate fmeini3-ml-assignment-3
    ```

Alternatively, you can install the latest version of the required packages manually from *PIP*:
`pip install numpy pandas scipy matplotlib seaborn scikit-learn yellowbrick`

## Run the code
Code is divided into multiple python scripts that loosely map the structure of the assignment:
- `python clustering.py` runs KMeans and EM clustering and generates plots for them (inside the *plots* folder)
- `python dimensionality.py` runs PCA, ICA, Randomized Projections, and SVD dimensionality reduction algorithms and generates plots for them (inside the *plots* folder)
- `python clustering_on_reduced_data.py` runs KMeans and EM clustering on the dataset with reduced dimensionality and generates plots for them (inside the *plots* folder)
- `python neural_networks.py` trains neural network classifiers on many versions of the Fashion MNIST dataset (base, reduced, augmented with clusters) and generates plots for them (inside the *plots* folder)
- The other scripts are used to transform and save data. It is not necessary to run them, since all the data is already present inside the *data* folder.