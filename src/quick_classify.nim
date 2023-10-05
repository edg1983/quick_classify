import strutils
import sequtils
import math
import json
import times
import random
import strformat
import quickclass/argparser
import quickclass/utils
import sets
import arraymancer
import tables
import zip/gzipfiles
import os

const VERSION="0.2"

const tmpl_html = staticRead("quickclass/template.html")

type Data_matrix = object
  labels*: seq[string]
  label_order*: Table[string, int]
  sids*: seq[string]
  mtx*: seq[seq[float32]]

type ForHtml = ref object
  text*: seq[string]
  nDims*: int
  dims: seq[seq[float32]]
  probs: seq[float32] # probability of maximum prediction
  group_label: string

proc get_labels_as_int(d: Data_matrix): seq[int] =
  result = newSeq[int](d.labels.len)
  for i, l in d.labels:
    result[i] = d.label_order[l]

proc read_data(d: var Data_matrix, filename: string, label_order: Table[string, int] = initTable[string, int]()) =
  log("INFO", fmt"Reading data from {filename}")
  let fs = (if filename.endsWith(".gz"): newGzFileStream(filename) else: newFileStream(filename))

  let header = fs.readLine().strip(chars = {' ', '\n', '#'}).split("\t")

  log("INFO", fmt"header: {header}")

  let sample_label_idx = header.find("label")
  if sample_label_idx == -1:
    log("ERROR", fmt"bad header! Cannot find 'label' column: {header}")
    quit QuitFailure
  
  let sample_id_idx = header.find("sample")
  if sample_id_idx == -1:
    log("ERROR", fmt"bad header! Cannot find 'sample' column: {header}")
    quit QuitFailure
  
  var mtx: seq[seq[float32]]
  var ancs: seq[string]

  var n = 0
  while not fs.atEnd():
    n += 1
    let line = fs.readLine().split("\t")
    if line.len != header.len:
      log("ERROR", fmt"bad line in {filename}: {line}")
      quit QuitFailure
    d.labels.add(line[sample_label_idx])
    d.sids.add(line[sample_id_idx])

    var idx = ancs.find(line[sample_label_idx])
    if idx == -1:
      ancs.add(line[sample_label_idx])
      idx = ancs.high
    discard d.label_order.hasKeyOrPut(line[sample_label_idx], idx)

    let float_line = line[2..line.high].map(proc(x: string): float32 = parseFloat(x))
    mtx.add(float_line)

  d.mtx = mtx
  log("INFO", fmt"read {n} entries from {filename}")    

proc main*() =

  echo(&"QUICK CLASSIFY VERSION {VERSION}")

  ### Parse command line arguments ###
  var opts = parseCmdLine()
  opts.logArgs()

  if opts.output_prefix.endswith("/"):
    opts.output_prefix &= "/quick_classify"
  if not opts.output_prefix.endswith("."):
    opts.output_prefix &= "."

  var 
    make_html = opts.make_html
    nHidden = parseInt(opts.nn_hidden_size)
    nn_test_samples: int
    model_json: JsonNode
  let
    nPCs = parseInt(opts.n_pcs)
    nEpochs = parseInt(opts.nn_epochs)
    batch_size = parseInt(opts.nn_batch_size)

  var perform_training = true
  if opts.model != "": perform_training = false

  var 
    train_preds_fh: File
    query_preds_fh: File 
    train_data: Data_matrix
    query_data: Data_matrix
    t_proj: Tensor[float32]
    q_proj: Tensor[float32]
    nDims: int
    nOut: int
  
  let
    train_preds_filename = opts.output_prefix & "predictions-train.tsv"
    query_preds_filename = opts.output_prefix & "predictions-query.tsv"
    html_filename = opts.output_prefix & "predictions.html"
  
  #Read the query data
  if not open(query_preds_fh, query_preds_filename, fmWrite):
    log("ERROR", &"couldn't open output file {query_preds_filename}")
    quit QuitFailure
  query_data.read_data(opts.query, train_data.label_order)

  #Check if we can create HTML when needed
  var fh_html: File
  if make_html:
    if not fh_html.open(html_filename, fmWrite):
      log("ERROR", &"couldn't open output file: {html_filename}")
      quit QuitFailure

  #If we are performing training, read the training data
  #Otherwise, get model config from JSON
  if perform_training:
    if not open(train_preds_fh, train_preds_filename, fmWrite):
      log("ERROR", &"couldn't open output file {train_preds_filename}")
      quit QuitFailure
    train_data.read_data(opts.train)
    nOut = train_data.sids.len
  else:
    for f in @["/fc1.weight.npy", "/fc1.bias.npy", "/classifier.weight.npy", "/classifier.bias.npy", "/model.json"]:
      if not fileExists(opts.model & f):
        log("ERROR", &"Unable to load model data: file {opts.model & f} does not exist")
        quit QuitFailure
    model_json = parseFile(opts.model & "/model.json")
    nOut = model_json["nOut"].getInt
    nHidden = model_json["nHidden"].getInt

  #Convert query matrix to tensor
  var Q = query_data.mtx.toTensor() #.transpose
  
  #Now get the features dimension
  #This is equal to the number of PCs if specified, otherwise number of features in the query data
  #We assume train and query data have the same features
  if nPCs > 0:
    nDims = nPCs
  else:
    nDims = Q.shape[1]

  #We turn off HTML report when N features is too large
  if make_html and (nDims > 50):
    log("WARNING", &"Maximum allowed features in the HTML is 50. You have {nDims} features (input or PCs). HTML will not be generated.")
    make_html = false

  ### PREPARE THE MODEL ###
  let
    ctx = newContext Tensor[float32]
  var
    t_probs: Tensor[float32]
    q_probs: Tensor[float32]

  network PredictionNet:
    layers:
      #x: Input([1, t_proj.shape[1]])
      fc1: Linear(nDims, nHidden)
      classifier: Linear(nHidden, nOut)

    forward x:
      x.fc1.relu.classifier

  if perform_training:
    ### PERFORM MODEL TRAINING ###
    var
      T = train_data.mtx.toTensor() #.transpose
      Y = train_data.get_labels_as_int.toTensor()
      t0 = cpuTime()

    if nEpochs < 500:
      log("ERROR", &"nEpochs set to {nEpochs}. Must be >= 500")
      quit QuitFailure

    #PC dimensionality reduction if configured
    if nPCs > 0:
      var res = T.pca_detailed(n_components = nPCs)
      log("INFO", fmt"Reduced dataset to shape {res.projected.shape}: {elapsed_time(t0)}")
      t_proj = T * res.components
      q_proj = Q * res.components
      if opts.save_model:
        res.components.write_npy(&"{opts.output_prefix}model/pc_components.npy")
    else:
      t_proj = T
      q_proj = Q
              
    if opts.nn_test_samples.contains('.'):
      nn_test_samples = (t_proj.shape[0].float * parseFloat(opts.nn_test_samples)).floor.int
    else:
      nn_test_samples = parseInt(opts.nn_test_samples)
    log("INFO", fmt"N samples for test convergence set to {nn_test_samples}")
    
    var
      X = ctx.variable t_proj

    log("INFO", &"training data shape: {X.value.shape}")

    var model = ctx.init(PredictionNet)

    #Function to save the model to disk
    proc save_model(network: PredictionNet[float32], model_folder: string, nHidden: int, nOut: int, labels: Table[string, int], nPCs: int, pc_res: PCA_Detailed) =
      var ordered_labels = newSeq[string](labels.len)
      for k, v in labels:
        ordered_labels[v] = k
      createDir(model_folder)
      network.fc1.weight.value.write_npy(&"{model_folder}/fc1.weight.npy")
      network.fc1.bias.value.write_npy(&"{model_folder}/fc1.bias.npy")
      network.classifier.weight.value.write_npy(&"{model_folder}/classifier.weight.npy")
      network.classifier.bias.value.write_npy(&"{model_folder}/classifier.bias.npy")
      var model_json = %* {"nHidden": nHidden, "nOut": nOut, "labels": ordered_labels, "nPCs": nPCs}
      var json_fh = open(&"{model_folder}/model.json", fmWrite)
      json_fh.write(model_json.pretty)
      json_fh.close()
      log("INFO", &"saved model to {model_folder}")
  
    let optim = model.optimizer(SGD, learning_rate = 0.01'f32)

    t0 = cpuTime()
    # range of data
    var proj_range = t_proj[_, 0].max() -  t_proj[_, 0].min()
    var rand_scale = proj_range / 5.0'f32

    #Actual model training loop
    log("INFO", "Start model training")
    for epoch in 0..nEpochs:

      # adds random-ness scaled inversely by proportion variance explained.
      var r = randomTensor[float32](t_proj.shape[0], t_proj.shape[1], rand_scale) -. (rand_scale / 2'f32)
      #r = r /. erv
      #echo r.mean(axis=0)
      var t_proj_r = t_proj +. r
      #echo t_proj_r.mean(axis=0)
      X = ctx.variable t_proj_r

      for batch_id in 0..<X.value.shape[0] div batch_size:

        let offset = batch_id * batch_size
        if offset > X.value.shape[0] - nn_test_samples:
          break

        let offset_stop = min(offset + batch_size,  X.value.shape[0] - nn_test_samples)
        let x = X[offset ..< offset_stop, _]
        let y = Y[offset ..< offset_stop]

        let
          clf = model.forward(x)
          loss = clf.sparse_softmax_cross_entropy(y)

        loss.backprop()
        optim.update()

      if epoch mod 500 == 0:
        ctx.no_grad_mode:
          let
            clf = model.forward(X[X.value.shape[0] - nn_test_samples..<X.value.shape[0], _])
            y_pred = clf.value.softmax.argmax(axis=1).squeeze
            y = Y[X.value.shape[0] - nn_test_samples..<X.value.shape[0]]
            loss = clf.sparse_softmax_cross_entropy(y).value.unsafe_raw_offset[0]
            accuracy = accuracy_score(y_pred, y)
        log("INFO", fmt"Epoch:{epoch}. loss: {loss:.5f}. accuracy on unseen data: {accuracy:.3f}.  total-time: {elapsed_time(t0)}")
        if epoch >= 800 and ((loss < 0.005 and accuracy > 0.98) or (accuracy >= 0.995 and loss < 0.025)):
          log("INFO", fmt"breaking with trained model at this accuracy and loss")
          break
    log("INFO", fmt"Finished training model. total-time: {elapsed_time(t0)}")

    # save the model if requested
    if opts.save_model:
      let model_folder = opts.output_prefix & "model"
      model.save_model(model_folder, nHidden, nOut, train_data.label_order, nPCs, res)

    # store the predictions
    ctx.no_grad_mode:
      t_probs = model.forward(X).value.softmax #.argmax(axis=1).squeeze
    
    q_probs = model.forward(ctx.variable q_proj).value.softmax

  else:
    ### LOAD A PRE-TRAINED MODEL ###
    proc load_model(ctx: Context[Tensor[float32]], model_folder: string): PredictionNet[float32] =
      result.fc1.weight = ctx.variable(read_npy[float32](&"{model_folder}/fc1.weight.npy"), requires_grad = true)
      result.fc1.bias   = ctx.variable(read_npy[float32](&"{model_folder}/fc1.bias.npy"), requires_grad = true)
      result.classifier.weight = ctx.variable(read_npy[float32](&"{model_folder}/classifier.weight.npy"), requires_grad = true)
      result.classifier.bias   = ctx.variable(read_npy[float32](&"{model_folder}/classifier.bias.npy"), requires_grad = true)
      log("INFO", &"loaded model from {model_folder}")

    var model = load_model(ctx, opts.model)
    let nPCs_in_saved_model = model_json["nPCs"].getInt
    if nPCs_in_saved_model > 0:
      log("INFO", &"performing PCA on query data projecting on {nPCs_in_saved_model} PCs from the saved model")
      let pc_components = read_npy[float32](&"{opts.model}/pc_components.npy")
      q_proj = Q * pc_components
    else:
      q_proj = Q
    q_probs = model.forward(ctx.variable q_proj).value.softmax
  
  var header = @["#sample_id", "predicted_label", "given_label"]
  # Load ordered labels from the JSON
  var labels: Table[string, int]
  if perform_training:
    labels = train_data.label_order
  else:
    var i = 0
    for l in model_json["labels"].items:
      labels[l.getStr] = i
      i += 1
  
  ### CREATE THE OUTPUT FILES ###
  # maintain order of labels
  # labels are converted to int for predictions so we need to map back
  var inv_orders = newSeq[string](labels.len)
  header.setLen(header.len + labels.len)
  for k, v in labels:
    inv_orders[v] = k
    header[3 + v] = k & "_prob"
  for ip in 0..<nPCs:
    header.add("PC" & $(ip + 1))

  # Initi tables for HTML
  var lhtmls = initTable[string, ForHtml]()
  var qhtmls = initTable[string, ForHtml]()

  # Save prediction results on training if we are not using a pre-trained model
  if perform_training:
    train_preds_fh.write_line(join(header, "\t"))
    let t_pred = t_probs.argmax(axis=1).squeeze
    for i, s in train_data.sids:
      # The order here is sampleID, predicted_label, given_label
      var line = @[s, inv_orders[t_pred[i]], train_data.labels[i]]
      for j in 0..<labels.len:
        line.add(formatFloat(t_probs[i, j], ffDecimal, precision=4))

      if make_html:
        var lhtml = lhtmls.mgetOrPut(train_data.labels[i], ForHtml(group_label: train_data.labels[i], nDims: nDims, dims: newSeq[seq[float32]](nDims)))
        lhtml.text.add(&"sample:{s} label-probability: {t_probs[i, _].max}")
        for j in 0..<nDims: lhtml.dims[j].add(t_proj[i, j])

      for j in 0..<nPcs:
        line.add(formatFloat(t_proj[i, j], ffDecimal, precision=4))

      train_preds_fh.write_line(join(line, "\t"))

    train_preds_fh.close
    log("INFO", &"wrote train predictions to {train_preds_filename}")

  # Save prediction results on query data
  let q_pred = q_probs.argmax(axis=1).squeeze
  for i, s in query_data.sids:
    let group_label = inv_orders[q_pred[i]]
    var line = @[s, inv_orders[q_pred[i]], ""]
    for j in 0..<labels.len:
      line.add(formatFloat(q_probs[i, j], ffDecimal, precision=4))

    var qhtml: ForHtml
    if make_html:
      qhtml = qhtmls.mgetOrPut(group_label, ForHtml(group_label: group_label, nDims: nDims, dims: newSeq[seq[float32]](nDims)))
      qhtml.text.add(&"sample:{s} label-probability: {q_probs[i, _].max:.4f}")
      for j in 0..<nDims: qhtml.dims[j].add(q_proj[i, j])

    for j in 0..<nPcs:
      line.add(formatFloat(q_proj[i, j], ffDecimal, precision=4))
    
    query_preds_fh.write_line(join(line, "\t"))

  query_preds_fh.close
  log("INFO", &"wrote query predictions to {query_preds_filename}")

  # Write HTML if requested
  if make_html:
    var htmls = tmpl_html.split("<BACKGROUND_JSON>")
    fh_html.write(htmls[0])
    fh_html.write_line(%* lhtmls)
    htmls = htmls[1].split("<QUERY_JSON>")
    fh_html.write(htmls[0])
    fh_html.write_line(%* qhtmls)
    fh_html.write(htmls[1])
    fh_html.close()
    log("INFO", &"wrote html file to {html_filename}")

when isMainModule:
  main()
