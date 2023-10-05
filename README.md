# Quick classifier

This is a simple classifier build in Nim language with Arraymancer. Given a set of training and a query, it trains a NN model and then use this to predict query labels. It can also perform PC dimensionality reduction and generate a HTML report of the predictions.

In case of genetic data, you will need to perform PC dimensionality reduction before using plink or similar, save results to a tab-separated file and then use this pre-computed PCs as input for this tool.

## Usage

```bash
QUICK CLASSIFY VERSION 0.2
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
  --save_model               save the trained model and eventually PC loadings for future use
  --model=MODEL              load a previously optimized model from this file (default: )
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

At the moment, pre-processing of input is limited. Thus, **test and train data must contain the same features (columns) in the same order**.

When `--n_pcs` is set to zero (default) the tool will use the input data directly (all columns strating from the 3rd), otherwise it will first perform PC dimensionality reduction generating N PCs and use these for model training and prediction.

## Output files

Based on the configured output_prefix the tool generates:

- `<output_prefix>.predictions-train.tsv` containing the predictions for the training dataset
- `<output_prefix>.predictions-query.tsv` containing the predictions for the query dataset

Both files contains the sample id, the given label, the predicted label and the probability estimstes for each label.

When `--make_html` is set the tool also generates a HTML report `<output_prefix>.predictions.html` containing

When `--save_model` is set the tool also generates a folder `<output_prefix>.model` containing data of the trained model and eventually the PC loadings.

## Re-use trained models

If you have a model folder containing model data from a previou run, you can use `--model <model_folder>` to load the model and use it for prediction. In this case, the tool will ignore all other parameters and use the model as is.

**NB.** In this case the query dataset must contain the same features (columns) as the training dataset in the same order. The features used in training are stored in the model folder in model.json.

If the training process involved PC dimensionality reduction, the query dataset will be automatically projected in the same PC space before prediction.

## Future improvements

- [ ] allow to subset columns in input files using column names or index
- [ ] when both train and test are given automatically subset to common columns
