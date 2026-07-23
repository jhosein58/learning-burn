# Serialization Precision and File Overhead (Example 13.2)

A practical investigation into serialization precision in **Burn** and **PyTorch**, highlighting the trade-offs between float precision ($f32$ vs $f16$), inference accuracy, and actual file sizes on disk.

## The Core Concept
When saving a model, the choice of "Recorder" dictates the numeric precision of the stored weights:
- **Full Precision (`f32`):** Lossless serialization (`NamedMpkFileRecorder<FullPrecisionSettings>`). Predictions match the trained state bit-for-bit.
- **Half Precision (`f16`):** Slightly lossy, compact serialization (`CompactRecorder`). Weights are rounded, causing minor differences in downstream inference values.

## The Format Overhead Paradox
While $f16$ is widely known to halve model size, saving a tiny model (e.g., 2 parameters) in $f16$ can actually yield a *larger* file than $f32$. 

This happens because serialization formats (like MessagePack) have a fixed metadata overhead (field names, type tags, structures). For microscopic models, this metadata dominates the file size. Once a model scales up to thousands or millions of parameters, the raw weight savings overwhelm the metadata overhead, and the file size is successfully halved.

## Why Use $f16$?
The primary benefit of $f16$ is not disk space, but rather **hardware performance**. By halving the memory footprint, it reduces memory bandwidth bottlenecks during GPU inference, speeding up data transfers to the execution units.
