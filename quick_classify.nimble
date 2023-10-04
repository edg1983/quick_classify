# Package

version       = "0.1.0"
author        = "edoardo.giacopuzzi"
description   = "Quickly classify samples with a multi-layer NN"
license       = "MIT"
srcDir        = "src"
bin           = @["quick_classify"]


# Dependencies

requires "nim >= 1.6.4", "argparse >= 3.0.0", "arraymancer == 0.7.15", "nimhdf5 == 0.5.10"
