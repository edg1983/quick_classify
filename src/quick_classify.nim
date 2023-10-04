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
import nimhdf5

const VERSION="0.1"

const tmpl_html = staticRead("quickclass/template.html")

type Data_matrix = object
  labels*: seq[string]
  label_order*: Table[string, int]
  sids*: seq[string]
  mtx*: seq[seq[float32]]

type ForHtml = ref object
  text*: seq[string]
  nPCs*: int
  pcs: seq[seq[float32]]
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

  var opts = parseCmdLine()
  opts.logArgs()

  if opts.output_prefix.endswith("/"):
    opts.output_prefix &= "/quick_classify"
  if not opts.output_prefix.endswith("."):
    opts.output_prefix &= "."

  var train_preds_fh: File
  var query_preds_fh: File
  let
    train_preds_filename = opts.output_prefix & "predictions-train.tsv"
    query_preds_filename = opts.output_prefix & "predictions-query.tsv"
    html_filename = opts.output_prefix & "predictions.html"
  if not open(train_preds_fh, train_preds_filename, fmWrite):
    log("ERROR", &"couldn't open output file {train_preds_filename}")
    quit QuitFailure
  if not open(query_preds_fh, query_preds_filename, fmWrite):
    log("ERROR", &"couldn't open output file {query_preds_filename}")
    quit QuitFailure

  var 
    train_data: Data_matrix
    query_data: Data_matrix

  train_data.read_data(opts.train)
  query_data.read_data(opts.query, train_data.label_order)

  var
    nPCs = parseInt(opts.n_pcs)
    nEpochs = parseInt(opts.nn_epochs)
    T = train_data.mtx.toTensor() #.transpose
    Q = query_data.mtx.toTensor() #.transpose
    Y = train_data.get_labels_as_int.toTensor()
    t0 = cpuTime()

  var fh_html: File
  if opts.make_html:
    if not fh_html.open(html_filename, fmWrite):
      log("ERROR", &"couldn't open output file: {html_filename}")
      quit QuitFailure

  if nEpochs < 500:
    log("ERROR", &"nEpochs set to {nEpochs}. Must be >= 500")
    quit QuitFailure

  var 
    t_proj: Tensor[float32]
    q_proj: Tensor[float32]
  if nPCs > 0:
    var res = T.pca_detailed(n_components = nPCs)
    log("INFO", fmt"Reduced dataset to shape {res.projected.shape}: {elapsed_time(t0)}")
    t_proj = T * res.components
    q_proj = Q * res.components
  else:
    t_proj = T
    q_proj = Q
        
  let
    ctx = newContext Tensor[float32]
    nHidden = parseInt(opts.nn_hidden_size)
    nOut = train_data.sids.len
    nn_test_samples = parseInt(opts.nn_test_samples)
  
  var
    X = ctx.variable t_proj

  log("INFO", &"training data shape: {X.value.shape}")

  network PredictionNet:
    layers:
      #x: Input([1, t_proj.shape[1]])
      fc1: Linear(t_proj.shape[1], nHidden)
      classifier: Linear(nHidden, nOut)

    forward x:
      x.fc1.relu.classifier
  
  proc save_model(network: PredictionNet, h5df_file: string) =
    var h5df = H5open(h5df_file, "rw")
    h5df.write(fc1.weight.value, group="fc1", name="weight")
    h5df.write(fc1.bias.value, group="fc1", name="bias")
    h5df.write(classifier.weight.value, group="classifier", name="weight")
    h5df.write(classifier.bias.value, group="classifier", name="bias")
    h5df.close()

  proc load_model(ctx: Context[Tensor[float32]], h5df_file: string): PredictionNet =
    result.fc1.weight = ctx.variable(read_hdf5[float32](h5df_file, group="fc1", name="weight"), requires_grad = true)
    result.fc1.bias   = ctx.variable(read_hdf5[float32](h5df_file, group="fc1", name="bias"), requires_grad = true)
    result.classifier.weight = ctx.variable(read_hdf5[float32](h5df_file, group="classifier", name="weight"), requires_grad = true)
    result.classifier.bias   = ctx.variable(read_hdf5[float32](h5df_file, group="classifier", name="bias"), requires_grad = true)

  let
    model = ctx.init(PredictionNet)
    optim = model.optimizer(SGD, learning_rate = 0.01'f32)

  if opts.model != "":
    model = load_model(ctx, opts.model)
    log("INFO", &"loaded model from {opts.model}")
  else:
    var batch_size: int

    if opts.nn_batch_size.contains('.'):
      batch_size = (t_proj.shape[0].float * parseFloat(opts.nn_batch_size)).floor.int
    else:
      batch_size = parseInt(opts.nn_batch_size)

    log("INFO", fmt"batch_size for testing set to {batch_size}")

    t0 = cpuTime()
    # range of data in first PC
    var proj_range = t_proj[_, 0].max() -  t_proj[_, 0].min()
    var rand_scale = proj_range / 5.0'f32
    #echo "rand_scale:", rand_scale

    log("INFO", "Start model training")
    # train the model
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
      let model_file = &"{opts.output_prefix}model.h5df"
      model.save_model(model_file)
      log("INFO", fmt"Model saved to {model_file}")

  ctx.no_grad_mode:
    let t_probs = model.forward(X).value.softmax #.argmax(axis=1).squeeze

  let
    q_probs = model.forward(ctx.variable q_proj).value.softmax
    q_pred = q_probs.argmax(axis=1).squeeze
    t_pred = t_probs.argmax(axis=1).squeeze

  var header = @["#sample_id", "predicted_label", "given_label"]
  #var inv_orders = newSeq[string](unique_labels.len)
  # maintain order of ancestries
  #header.setLen(header.len + unique_labels.len)
  var inv_orders = newSeq[string](train_data.label_order.len)
  # maintain order of ancestries
  header.setLen(header.len + train_data.label_order.len)
  for k, v in train_data.label_order:
    inv_orders[v] = k
    header[3 + v] = k & "_prob"
  for ip in 0..<nPCs:
    header.add("PC" & $(ip + 1))

  train_preds_fh.write_line(join(header, "\t"))

  var lhtmls = initTable[string, ForHtml]()
  var qhtmls = initTable[string, ForHtml]()

  for i, s in train_data.sids:
    # The order here is sampleID, predicted_label, given_label
    var line = @[s, inv_orders[t_pred[i]], train_data.labels[i]]
    for j in 0..<train_data.label_order.len:
      line.add(formatFloat(t_probs[i, j], ffDecimal, precision=4))

    if opts.make_html:
      var lhtml = lhtmls.mgetOrPut(train_data.labels[i], ForHtml(group_label: train_data.labels[i], nPCs: nPCs, pcs: newSeq[seq[float32]](nPCs)))
      lhtml.text.add(&"sample:{s} label-probability: {t_probs[i, _].max}")

    for j in 0..<nPcs:
      line.add(formatFloat(t_proj[i, j], ffDecimal, precision=4))
    train_preds_fh.write_line(join(line, "\t"))

  train_preds_fh.close
  log("INFO", &"wrote train predictions to {train_preds_filename}")

  for i, s in query_data.sids:
    let group_label = inv_orders[q_pred[i]]
    var line = @[s, inv_orders[q_pred[i]], ""]
    for j in 0..<train_data.label_order.len:
      line.add(formatFloat(q_probs[i, j], ffDecimal, precision=4))

    var qhtml: ForHtml
    if opts.make_html:
      qhtml = qhtmls.mgetOrPut(group_label, ForHtml(group_label: group_label, nPCs: nPCs, pcs: newSeq[seq[float32]](nPCs)))
      qhtml.text.add(&"sample:{s} label-probability: {q_probs[i, _].max:.4f}")

    for j in 0..<nPcs:
      line.add(formatFloat(q_proj[i, j], ffDecimal, precision=4))
      if opts.make_html: qhtml.pcs[j].add(q_proj[i, j])
    query_preds_fh.write_line(join(line, "\t"))

  query_preds_fh.close
  log("INFO", &"wrote query predictions to {query_preds_filename}")

  if opts.make_html:
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
