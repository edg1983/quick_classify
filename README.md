# Quick classifier

This is a simple classifier build in Nim language with Arraymancer. Given a set of training and a query, it trains a NN model and then use this to predict query labels. It can also perform PC dimensionality reduction and generate a HTML report of the predictions.

In case of genetic data, you will need to perform PC dimensionality reduction before using plink or similar, save results to a tab-separated file and then use this pre-computed PCs as input for this tool.

## Usage

```bash
QUICK CLASSIFY VERSION 0.1
make a quick classifier and classify examples

Usage:
  quick_classify [options] 

Options:
  -h, --help
  -o, --output_prefix=OUTPUT_PREFIX
                             prefix for output files (default: quick_classify)
  -t, --train=TRAIN          separate tsv file containing training set
  -q, --query=QUERY          input file containing new data to predict
  --make_html                generate HTML report of predictions results. Not suggested with N dimensions is > 20
  --n_pcs=N_PCS              number of principal components to use in the reduced dataset (default: 0)
  --nn_epochs=NN_EPOCHS      number of epochs for model training (default: 10000)
  --nn_hidden_size=NN_HIDDEN_SIZE
                             shape of hidden layer in neural network (default: 16)
  --nn_batch_size=NN_BATCH_SIZE
                             batch size fo training neural network (default: 32)
  --nn_test_samples=NN_TEST_SAMPLES
                             number or fraction of labeled samples to test for NN convergence (default: 100)
```

## Input files

At the moment the tool only accept tab-separated files as input for train and query dataset. The first 2 column must contain sample id and labels and must be named `sample` and `label`. The label column is empty for the query dataset. All other columns are used as features, there is no option to subset columns at the moment.

When `--n_pcs` is set to zero (default) the tool will use the input data directly (all columns strating from the 3rd), otherwise it will first perform PC dimensionality reduction generating N PCs and use these for model training and prediction.

## Output files

Based on the configure output_prefix the tool generates:

- `<output_prefix>.predictions-train.tsv` containing the predictions for the training dataset
- `<output_prefix>.predictions-query.tsv` containing the predictions for the query dataset

Both files contains the sample id, the given label, the predicted label and the probability estimstes for each label.

When `--make_html` is set the tool also generates a HTML report `<output_prefix>.predictions.html` containing:

## Future improvements

- [ ] be able to save trained models and use them to predict new data without retraining the model.
- [ ] allow to subset columns in input files using column names or index
