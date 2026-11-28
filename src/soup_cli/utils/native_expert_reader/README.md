# Native expert reader

This C++17 extension reads packed expert records from shard file descriptors into
a bounded host ring. Soup ships the sources for explicit builds. The extension is
available for direct use after building.

## Build on Linux

Use the Python environment that will import the result, with Soup, PyTorch and
setuptools installed, plus a C++17 compiler. Run from an installed Soup
environment or an editable checkout.

```bash
python -m soup_cli.utils.native_expert_reader.build_ext build_ext \
    --build-lib /tmp/soup-native --build-temp /tmp/soup-native/obj
export PYTHONPATH="/tmp/soup-native${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch, _native_expert_stream_async'
```

Import PyTorch first so it loads the shared libraries for the extension. The
compiled module exports `PackedStreamAsync`. The CPU build needs no CUDA toolkit.

To also export `decode_nf4_bf16_pair` on Linux, add `--with-cuda-nf4` to the
build command. Use a CUDA-enabled PyTorch install and a matching toolkit with
`nvcc`. The decoder preserves BF16 subnormals by building with
`--ftz=false`. Give each build variant a separate output and object directory
if using both. The macOS source uses the platform's file timestamps for CPU builds.
Windows builds require a separate toolchain check.

If the build reports `CUDA 13.0 and glibc 2.43 disagree about rsqrt`, use a
compatible host toolchain or follow the diagnostic for the two declarations in
the CUDA toolkit header. The check reads the header and leaves it intact.

## Direct use

`PackedStreamAsync` duplicates registered file descriptors. Drive each instance
from a single consumer thread. Supply a writable host buffer of `packed_bytes`
for each read and keep it alive until the read returns. Consume results in
sequence order and call `finish_sequence()` when stopping early or handling a read
error as well as after a complete sequence. Call `close()` once finished with the
stream. The optional decoder takes raw CUDA pointers. Keep the buffers alive
until the supplied stream completes.
