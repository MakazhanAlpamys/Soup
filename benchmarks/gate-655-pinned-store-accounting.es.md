# Contabilidad de memoria del host fijada (#655)
> 🇪🇸 Traducción al español de [`gate-655-pinned-store-accounting.md`](gate-655-pinned-store-accounting.md). Ante cualquier duda, manda el original en inglés.

Medido por [Tristan Grech](https://github.com/tristangrech) el 2026-09-10 UTC.
El [comentario original del issue](https://github.com/MakazhanAlpamys/Soup/issues/655#issuecomment-5623582962)
se reproduce a continuación sin modificaciones, incluida la deriva del contador del host, siguiendo la
[solicitud del mantenedor](https://github.com/MakazhanAlpamys/Soup/issues/655#issuecomment-5629542226).

Para [#655](https://github.com/MakazhanAlpamys/Soup/issues/655), estas mediciones
respaldan mantener la comprobación previa `statvfs` de `/dev/shm` fuera de la ruta de pinned-store:
las dos asignaciones pinned aumentaron `RssShmem` exactamente 4 GiB, mientras el uso del
montaje no cambió. Por lo tanto, el espacio libre del montaje no es un límite de capacidad
para estas asignaciones en la pila probada, ni sustituye las comprobaciones de margen
suficiente en el host y el cgroup. Esto no establece el comportamiento en otras pilas ni garantiza
que quepa una ejecución completa en streaming.

## Registro de medición original

La predicción de contabilidad se reproduce en un contenedor RunPod Linux/CUDA. Dos ejecuciones pinned en procesos nuevos añadieron exactamente **4 GiB a `RssShmem`**, sin cambios en `RssAnon` ni en el uso de `/dev/shm`. El control pageable añadió exactamente **4 GiB a `RssAnon`**, sin cambios en `RssShmem`.

Entorno: NVIDIA L4; controlador `570.195.03`; PyTorch `2.9.1+cu128`; runtime de CUDA `12.8`; Python `3.12.3`; contenedor Ubuntu 24.04.3; kernel del host Linux `6.14.0-33-generic`. Mediciones tomadas el 2026-09-10 UTC. No se configuró `PYTORCH_ALLOC_CONF` ni `PYTORCH_CUDA_ALLOC_CONF`.

Cada ejecución usó un proceso Python nuevo. Después de inicializar CUDA y hacer un calentamiento pinned de 1 MiB, asignó `4 * 1024**3` bytes de memoria del host `torch.uint8` y tocó cada página con `fill_(1)`. `is_pinned()` fue true en ambas ejecuciones pinned y false en el control. El proceso leyó directamente su propio `/proc/self/status` y `smaps_rollup`.

Todos los números siguientes están en **KiB, antes → después**. La asignación es de 4,194,304 KiB.

| Contador | Ejecución pinned 1 | Ejecución pinned 2 | Control pageable |
|---|---:|---:|---:|
| Proceso `RssAnon` | 283648 → 283648 | 282624 → 282624 | 283648 → 4477952 |
| Proceso `RssShmem` | 10240 → 4204544 | 10240 → 4204544 | 10240 → 10240 |
| Proceso `VmLck` | 0 → 0 | 0 → 0 | 0 → 0 |
| Host `Shmem` | 381632 → 4575968 | 384296 → 4578760 | 384288 → 384456 |
| Host `Mlocked` | 27540 → 27540 | 27540 → 27572 | 27540 → 27540 |
| Host `Unevictable` | 30612 → 30612 | 30612 → 30644 | 30612 → 30612 |
| Uso de `/dev/shm` | 0 → 0 | 0 → 0 | 0 → 0 |

`smaps_rollup.Pss_Shmem` aumentó independientemente exactamente 4 GiB en cada ejecución pinned y permaneció sin cambios en el control. `memory.stat.shmem` de cgroup v2 también aumentó exactamente 4 GiB para la asignación pinned. No hubo eventos OOM del cgroup.

El `memory.max` del contenedor era 93,999,996,928 bytes; el montaje `/dev/shm` era de 47,000,002,560 bytes. El script comprobó el margen disponible del host y del cgroup antes de asignar. Los contadores de todo el host pueden incluir actividad de otros tenants, por lo que los pequeños cambios de fondo en esas columnas no deben atribuirse a esta asignación.

Esto respalda la contabilidad de memoria compartida que predecía el issue en esta pila: la asignación pinned no consume el montaje `/dev/shm`. No establece la implementación interna del controlador, no demuestra el comportamiento en todos los controladores/kernel ni mide la posibilidad de recuperación de memoria. No ejecuté la prueba opcional de streaming cerca del 55% de la RAM del host ni provoqué un OOM.

El pod temporal se terminó. La sonda exacta siguiente puede ejecutarse dos veces con `--kind pinned` y una vez con `--kind pageable`; cada invocación emite su entorno y los contadores sin procesar antes/después como JSON. SHA-256 de la sonda: `44bba1b76a9b41e927fe59666104bd44fcce63288f4af48887736118f7111c11`.


<details>
<summary>Sonda exacta usada (guardar como probe.py)</summary>

```python
"""Soup #655: bounded pinned-host-memory accounting probe; one fresh process/run."""
import argparse
import gc
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

GIB = 1024**3


def kb_fields(path, wanted):
    result = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            result[key] = int(value.split()[0]) * 1024
    return result


def cgroup_snapshot():
    roots = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")]
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if ".." in Path(relative).parts:
            continue
        if not controllers:
            roots.append(Path("/sys/fs/cgroup") / relative.lstrip("/"))
        elif "memory" in controllers.split(","):
            roots.append(Path("/sys/fs/cgroup/memory") / relative.lstrip("/"))
    results = {}
    for root in dict.fromkeys(roots):
        for filename in ("memory.current", "memory.max", "memory.limit_in_bytes",
                         "memory.usage_in_bytes", "memory.stat", "memory.events"):
            path = root / filename
            if path.is_file():
                results[str(path)] = path.read_text().strip()
    return results


def snapshot():
    shm = os.statvfs("/dev/shm")
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "process_bytes": kb_fields("/proc/self/status", {
            "VmRSS", "VmLck", "RssAnon", "RssFile", "RssShmem"}),
        "smaps_rollup_bytes": kb_fields("/proc/self/smaps_rollup", {
            "Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Locked"}),
        "host_bytes": kb_fields("/proc/meminfo", {
            "MemTotal", "MemAvailable", "Shmem", "Mlocked", "Unevictable"}),
        "dev_shm_bytes": {
            "total": shm.f_blocks * shm.f_frsize,
            "used": (shm.f_blocks - shm.f_bfree) * shm.f_frsize,
            "available": shm.f_bavail * shm.f_frsize,
        },
        "cgroup": cgroup_snapshot(),
    }


def check_headroom(before, allocation_bytes):
    headrooms = [before["host_bytes"]["MemAvailable"]]
    cgroup = before["cgroup"]
    pairs = (("memory.max", "memory.current"),
             ("memory.limit_in_bytes", "memory.usage_in_bytes"))
    found = False
    for path, value in cgroup.items():
        for limit_name, usage_name in pairs:
            if Path(path).name != limit_name:
                continue
            usage = cgroup.get(str(Path(path).with_name(usage_name)))
            if usage is not None:
                found = True
                if value != "max":
                    headrooms.append(int(value) - int(usage))
    if not found:
        raise RuntimeError("Cannot verify container memory headroom")
    if min(headrooms) < 2 * allocation_bytes:
        raise RuntimeError("Less than 2x allocation headroom; refusing probe")
    return headrooms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("pinned", "pageable"), required=True)
    parser.add_argument("--gib", type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    environment = {
        "kernel": platform.release(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"], text=True).strip(),
        "cgroup_membership": Path("/proc/self/cgroup").read_text().strip(),
        "allocator_environment": {key: os.environ.get(key) for key in (
            "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
        "os_release": Path("/etc/os-release").read_text().strip(),
    }
    torch.cuda.init()
    warmup = torch.empty(1024**2, dtype=torch.uint8, pin_memory=True)
    warmup.fill_(1)
    del warmup
    gc.collect()
    torch.cuda.synchronize()
    before = snapshot()
    allocation_bytes = args.gib * GIB
    headroom = check_headroom(before, allocation_bytes)
    start = time.monotonic()
    tensor = torch.empty(allocation_bytes, dtype=torch.uint8, device="cpu",
                         pin_memory=args.kind == "pinned")
    tensor.fill_(1)
    assert tensor.is_pinned() == (args.kind == "pinned")
    assert tensor[0].item() == 1 and tensor[-1].item() == 1
    time.sleep(0.25)
    after = snapshot()
    delta = {
        section: {key: after[section][key] - value
                  for key, value in before[section].items()}
        for section in ("process_bytes", "smaps_rollup_bytes", "host_bytes",
                        "dev_shm_bytes")
    }
    result = {
        "kind": args.kind, "allocation_bytes": allocation_bytes,
        "is_pinned": tensor.is_pinned(), "environment": environment,
        "headroom_bytes": headroom, "before": before, "after": after,
        "delta_bytes": delta, "duration_seconds": time.monotonic() - start,
        "note": "Process counters are local; host counters can include other tenants.",
    }
    del tensor
    gc.collect()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
```

```bash
python probe.py --kind pinned --gib 4
python probe.py --kind pinned --gib 4
python probe.py --kind pageable --gib 4
```
</details>
