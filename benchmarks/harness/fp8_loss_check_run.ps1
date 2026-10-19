<#
.SYNOPSIS
One arm of the FP8-source loss check (benchmarks/probe-rtx5070-moonlight-fp8-source.md).

.DESCRIPTION
Arm bf16 (A) trains Moonlight-16B-A3B from its bf16 checkpoint and reuses the fused NF4 layer
cache section 4.1 wrote (D:/soup-bench/soup-cache). Arm fp8 (B) trains from the FP8-block copy
and shards it into a new cache root (D:/soup-bench/fp8-loss-check/soup-cache, ~30.6 GB).

Run A first, then B no sooner than 10 minutes after A ended. AC power, lid open, no other GPU
work, the measurement hold announced. The script refuses to start when a pre-check fails:
src/ not at 25c876f6, soup_cli imported from elsewhere, < 10 GB RAM available, off AC, another
GPU compute process, the cache not in the state the rule expects (A reuse, B re-shard), B too
soon or too hot after A, or too little room on D: for B's cache.

It opens a second window running the RAM watchdog, then runs `soup train` with its console
copied to <arm>/train.log, appends "exit=<code>", and re-runs the cache check (both arms must
then reuse their cache).

.EXAMPLE
pwsh -NoProfile -ExecutionPolicy Bypass -File C:\Users\user\projects\soup-wt\moe-fp8\benchmarks\harness\fp8_loss_check_run.ps1 -Arm bf16
#>
param([Parameter(Mandatory = $true)][ValidateSet('bf16', 'fp8')][string]$Arm)
$ErrorActionPreference = 'Stop'

$Tree = 'C:\Users\user\projects\soup-wt\moe-fp8'
$Commit = '25c876f6552891173e1b4aa61046cfe6e49142cd'
$Harness = Join-Path $Tree 'benchmarks\harness'
$Bench = 'D:\soup-bench'
$Root = Join-Path $Bench 'fp8-loss-check'
$Run = Join-Path $Root $Arm
$Base = @{ bf16 = 'D:/soup-bench/models/Moonlight-16B-A3B'; fp8 = 'D:/soup-bench/models/Moonlight-16B-A3B-fp8block' }[$Arm]
$CacheRoot = @{ bf16 = 'D:/soup-bench/soup-cache'; fp8 = 'D:/soup-bench/fp8-loss-check/soup-cache' }[$Arm]
$Expect = @{ bf16 = 'reuse'; fp8 = 'reshard' }[$Arm]

Set-Location $Bench
foreach ($Item in 'output', 'experiments.db', 'train.log', 'watchdog.json', 'watchdog.jsonl') {
    # A re-run (section 2.4) moves the first attempt's files into a subfolder first.
    if (Test-Path (Join-Path $Run $Item)) { throw "$Run\$Item exists: this arm already ran" }
}
$env:PYTHONPATH = Join-Path $Tree 'src'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:COLUMNS = '160'
$env:SOUP_LAYER_STREAM_STRIPE_DIRS = $null
$env:SOUP_LAYER_STREAM_CACHE_DIR = $CacheRoot
$env:SOUP_DB_PATH = Join-Path $Run 'experiments.db'

# 1. Code: src/ is the commit's, and Python imports it from this tree.
git -C $Tree diff --quiet $Commit -- src
if ($LASTEXITCODE -ne 0) { throw "src/ differs from $Commit" }
if (git -C $Tree status --porcelain -- src) { throw 'untracked or modified files under src/' }
$env:CUDA_VISIBLE_DEVICES = '-1'
$Imported = python -c "import soup_cli; print(soup_cli.__file__)"
if ($Imported -ne (Join-Path $Tree 'src\soup_cli\__init__.py')) { throw "soup_cli imports from $Imported" }

# 2. Box: RAM, AC, no other GPU compute process.
$FreeGB = [double](python -c "import psutil; print(psutil.virtual_memory().available / 1e9)")
if ($FreeGB -lt 10) { throw "only $FreeGB GB of RAM available; the rule asks for 10" }
$OnAC = python -c "import psutil; b = psutil.sensors_battery(); print(b is None or b.power_plugged)"
if ($OnAC -ne 'True') { throw 'not on AC power' }
$env:CUDA_VISIBLE_DEVICES = $null
if (nvidia-smi --query-compute-apps=pid --format=csv,noheader) { throw 'another process is using the GPU' }
$Temp = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
if ($Arm -eq 'fp8') {
    $LogA = Join-Path $Root 'bf16\train.log'
    if (-not (Test-Path $LogA)) { throw 'arm A has not run' }
    $Idle = (Get-Date) - (Get-Item $LogA).LastWriteTime
    if ($Idle.TotalMinutes -lt 10) { throw "arm A ended $([int]$Idle.TotalMinutes) min ago; wait 10" }
    $TempA = [int](Get-Content (Join-Path $Root 'bf16\gpu-temp-before.txt'))
    if ($Temp -gt $TempA + 5) { throw "GPU at $Temp C, arm A started at $TempA C; let it cool" }
}
$Temp | Set-Content (Join-Path $Run 'gpu-temp-before.txt')

# 3. Cache: A must reuse section 4.1's cache, B must shard a new one (and fit with 10 GB spare).
$env:CUDA_VISIBLE_DEVICES = '-1'
python (Join-Path $Harness 'fp8_loss_check_cache.py') --base $Base --expect $Expect --json (Join-Path $Run 'cache-before.json')
if ($LASTEXITCODE -ne 0) { throw "cache check failed: expected $Expect" }
$Cache = Get-Content (Join-Path $Run 'cache-before.json') | ConvertFrom-Json
if ($Arm -eq 'fp8' -and $Cache.cache_volume_free_bytes -lt $Cache.reshard_estimate_bytes + 10e9) {
    throw 'D: has too little room for the new cache plus 10 GB'
}
$env:CUDA_VISIBLE_DEVICES = $null

# 4. The RAM watchdog, in its own window: it waits for the training process and follows it.
$Watch = "python '$Harness\fp8_loss_check_watchdog.py' --match '$Run\soup.yaml' --out '$Run' --min-free-gb 1.5"
Start-Process pwsh -ArgumentList '-NoProfile', '-NoExit', '-Command', $Watch

# 5. The run. Native stderr is copied too; PowerShell 7.2+ does not apply
# $ErrorActionPreference to it, and the exit code is read from $LASTEXITCODE.
$ErrorActionPreference = 'Continue'
$Log = Join-Path $Run 'train.log'
python -m soup_cli.cli train --config "$Run\soup.yaml" --trust-remote-code --yes 2>&1 | Tee-Object -FilePath $Log
$Code = $LASTEXITCODE
"exit=$Code" | Tee-Object -FilePath $Log -Append

# 6. After: the cache this arm used must now be reusable (A's unchanged, B's written).
$env:CUDA_VISIBLE_DEVICES = '-1'
python (Join-Path $Harness 'fp8_loss_check_cache.py') --base $Base --expect reuse --json (Join-Path $Run 'cache-after.json')
$env:CUDA_VISIBLE_DEVICES = $null
Write-Host "arm $Arm finished with exit code $Code"
