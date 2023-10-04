import argparse
import strformat
from ./utils import log

var p = newParser("quick classify"):
    help("make a quick classifier and classify examples")
    #option("--labels", help="file with ancestry labels")
    option("-o", "--output_prefix", help="prefix for output files", default=some("quick_classify"))
    option("-t", "--train", help="separate tsv file containing training set")
    option("-q", "--query", help="input file containing new data to predict", required=true)
    flag("--make_html", help="generate HTML report of predictions results. Not suggested with N dimensions is > 20")
    flag("--save_model", help="save the trained model and eventually PC loadings for future use")
    option("--model", help="load a previously optimized model from this file", default=some(""))
    option("--n_pcs", help="number of principal components to use in the reduced dataset", default=some("0"))
    option("--nn_epochs", help="number of epochs for model training", default=some("10000"))
    option("--nn_hidden_size", help="shape of hidden layer in neural network", default=some("16"))
    option("--nn_batch_size", help="batch size fo training neural network", default=some("32"))
    option("--nn_test_samples", help="number or fraction of labeled samples to test for NN convergence", default=some("100"))
    #arg("extracted", nargs= -1, help="$sample.somalier files for each sample. place labelled samples first followed by '++' then *.somalier for query samples")

proc parseCmdLine*(): ref =
    try:
        result = p.parse() 
    except ShortCircuit as e:
        if e.flag == "argparse_help":
            echo p.help
            quit QuitSuccess
    except UsageError:
        stderr.writeLine getCurrentExceptionMsg() 
        echo "Use --help for usage information"
        quit QuitSuccess

proc logArgs*(opts: ref) {.discardable.} =
    log("ARG", fmt"Train file: {opts.train}")
    log("ARG", fmt"Test file: {opts.query}")
    log("ARG", fmt"Output prefix: {opts.output_prefix}")
    log("ARG", fmt"Number of PCs: {opts.n_pcs}")
    log("ARG", fmt"NN epochs: {opts.nn_epochs}")
    log("ARG", fmt"NN hidden size: {opts.nn_hidden_size}")
    log("ARG", fmt"NN batch size: {opts.nn_batch_size}")
    log("ARG", fmt"NN test samples: {opts.nn_test_samples}")