import times
import strformat
import system
import math
import std/os

proc log* (level: string = "INFO", msg: string) {.discardable.} = 
    let t = now()
    let f = initTimeFormat("HH:mm:ss-fff")
    var time_string = format(t, f)
    let log_msg = fmt"[{time_string}] - {level}: {msg}"
    stderr.write(log_msg & "\n")

proc elapsed_time* (start_time: float): string =
    let interval = cpuTime() - start_time
    let s = floor(interval)
    let m = floor(((interval - s) * 1000))
    let time_interval = initDuration(seconds = int(s), milliseconds = int(m))
    result = $time_interval

proc update_with_glob*(files: var seq[string]) =
    var toadd = newSeqOfCap[string](256)
    for i in 0..<min(files.len, 10):
        if files[i] == "++":
            toadd.add(files[i])
            continue
        for w in files[i].walkFiles:
            toadd.add(w)

    if files.len > 10:
        files = toadd & files[10..files.high]
    else:
        files = toadd