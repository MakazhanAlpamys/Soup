<#
.SYNOPSIS
Pre-declared gate for opt-in CUDA graph decoding (benchmarks/gate-v0.76.0-cuda-graph-decode.md).

.DESCRIPTION
Runs the decode harness UNPINNED in fresh processes, A-B-B-A-B-A, for the full model and a
synthetic LoRA; a pinned A-B bridge pair to the experiment's published numbers; then
`soup infer` end to end over 24 varied prompts, A-B-B-A. A is the branch's merge-base with
origin/main, which has no flag. B is the branch, with --cuda-graphs.
Start it from a FOREGROUND PowerShell window and leave that window focused until it prints
"Gate runs complete." (about 40 minutes). Keep the laptop on AC with the lid open.

Prefill phase diagnostics are off (--prefill-phase-repeats 0): they were diagnostic-only in
the experiment, no gate criterion reads them, and they import an experiment file that is not
published here.

.EXAMPLE
pwsh -NoProfile -ExecutionPolicy Bypass -File .\cuda_graph_decode_gate.ps1 -Workspace 'C:\Users\user\Soup Experimental'
#>
param([Parameter(Mandatory = $true)][string]$Workspace)
$ErrorActionPreference = 'Stop'
. (Join-Path $Workspace 'experiments/env.ps1')
# env.ps1 points PYTHONPATH at the experiment's own clone. The harness takes --source
# instead, and each CLI arm below sets its own path.
$env:PYTHONPATH = ''
$Experiments = Join-Path $Workspace 'experiments'
Set-Location $Workspace
$Python = Join-Path $Workspace '.venv/Scripts/python.exe'
# Run the PUBLISHED harness bytes, not whatever else sits in the workspace.
Copy-Item (Join-Path $PSScriptRoot 'cuda_graph_decode_bench.py') (Join-Path $Experiments 'cg_bench_decode.py') -Force
Copy-Item (Join-Path $PSScriptRoot 'cuda_graph_decode_compare.py') (Join-Path $Experiments 'cg_compare_results.py') -Force
$Harness = 'experiments/cg_bench_decode.py'
$Compare = 'experiments/cg_compare_results.py'
$Verdict = Join-Path $PSScriptRoot 'cuda_graph_decode_verdict.py'
$BaseModel = 'experiments/models/Qwen2.5-1.5B-Instruct'
$LoraModel = 'experiments/models/qwen2.5-1.5b-synthetic-lora'
$Order = @(
    @{Arm = 'a1'; Source = 'experiments/cg-main'; Variant = 'baseline'},
    @{Arm = 'b1'; Source = 'experiments/cg-candidate'; Variant = 'production_cuda_graphs'},
    @{Arm = 'b2'; Source = 'experiments/cg-candidate'; Variant = 'production_cuda_graphs'},
    @{Arm = 'a2'; Source = 'experiments/cg-main'; Variant = 'baseline'},
    @{Arm = 'b3'; Source = 'experiments/cg-candidate'; Variant = 'production_cuda_graphs'},
    @{Arm = 'a3'; Source = 'experiments/cg-main'; Variant = 'baseline'}
)

function Invoke-HarnessRun {
    param([string]$Name, [string]$Source, [string]$Variant, [string]$Model,
          [string]$Reference, [int]$Affinity = -1)
    $Arguments = @($Harness, '--source', $Source, '--variant', $Variant, '--model', $Model,
        '--output', "experiments/$Name.json", '--warmups', '3', '--repeats', '7',
        '--phase-repeats', '0', '--prefill-phase-repeats', '0', '--telemetry')
    if ($Affinity -ge 0) { $Arguments += @('--cpu-affinity', "$Affinity") }
    if ($Reference) { $Arguments += @('--reference', $Reference) }
    Write-Output "Starting $Name ($Variant from $Source, affinity $Affinity)"
    & $Python @Arguments *> "experiments/$Name.log"
    if ($LASTEXITCODE -ne 0) {
        Get-Content "experiments/$Name.log" -Tail 40
        throw "Run $Name failed with exit code $LASTEXITCODE"
    }
}

foreach ($Scenario in @(@{Tag = 'full'; Model = $BaseModel}, @{Tag = 'lora'; Model = $LoraModel})) {
    $Reports = @()
    foreach ($Step in $Order) {
        $Name = "cg-gate-$($Scenario.Tag)-$($Step.Arm)"
        $Reference = if ($Step.Arm -eq 'a1') { '' } else { "experiments/cg-gate-$($Scenario.Tag)-a1.json" }
        Invoke-HarnessRun -Name $Name -Source $Step.Source -Variant $Step.Variant -Model $Scenario.Model -Reference $Reference
        $Reports += "experiments/$Name.json"
    }
    $ComparisonPath = "experiments/cg-gate-$($Scenario.Tag)-comparison.json"
    & $Python $Compare @Reports --output $ComparisonPath
    if ($LASTEXITCODE -ne 0) { throw "Comparison for $($Scenario.Tag) failed" }
    & $Python $Verdict harness $ComparisonPath @Reports --output "experiments/cg-gate-$($Scenario.Tag)-verdict.json"
    Write-Output "Verdict $($Scenario.Tag): exit $LASTEXITCODE (0 = pass)"
}

# Bridge to the experiment's pinned numbers. It uses its own reference, because the
# harness (correctly) rejects an affinity mismatch against an unpinned run.
Invoke-HarnessRun -Name 'cg-gate-pinned-a' -Source 'experiments/cg-main' -Variant 'baseline' -Model $BaseModel -Reference '' -Affinity 0
Invoke-HarnessRun -Name 'cg-gate-pinned-b' -Source 'experiments/cg-candidate' -Variant 'production_cuda_graphs' -Model $BaseModel -Reference 'experiments/cg-gate-pinned-a.json' -Affinity 0

# (d) soup infer end to end, load + compile included, unpinned, A-B-B-A.
$CliDir = Join-Path $Experiments 'cg-gate-cli'
New-Item -ItemType Directory -Force -Path $CliDir | Out-Null
& $Python (Join-Path $PSScriptRoot 'cuda_graph_decode_prompts.py') --count 24 --output (Join-Path $CliDir 'prompts.jsonl')
if ($LASTEXITCODE -ne 0) { throw 'Prompt generation failed' }
Set-Location $CliDir
$ModelAbsolute = Join-Path $Workspace $BaseModel
$Wall = [ordered]@{}
foreach ($Step in @(@{Arm = 'a1'; Source = 'cg-main'}, @{Arm = 'b1'; Source = 'cg-candidate'},
                    @{Arm = 'b2'; Source = 'cg-candidate'}, @{Arm = 'a2'; Source = 'cg-main'})) {
    $SourceRoot = Join-Path $Experiments $Step.Source
    $env:PYTHONPATH = Join-Path $SourceRoot 'src'
    $Imported = & $Python -c "import soup_cli; print(soup_cli.__file__)"
    if (-not ($Imported -like "$SourceRoot*")) { throw "arm $($Step.Arm) imported $Imported" }
    $Arguments = @('-m', 'soup_cli', 'infer', '--model', $ModelAbsolute, '--input', 'prompts.jsonl',
        '--output', "out-$($Step.Arm).jsonl", '--temperature', '0', '--max-tokens', '128')
    if ($Step.Source -eq 'cg-candidate') { $Arguments += '--cuda-graphs' }
    $Seconds = (Measure-Command { & $Python @Arguments *> "cli-$($Step.Arm).log" }).TotalSeconds
    if ($LASTEXITCODE -ne 0) { throw "CLI arm $($Step.Arm) failed; see cli-$($Step.Arm).log" }
    $Wall[$Step.Arm] = $Seconds
    Write-Output "CLI $($Step.Arm): $Seconds s"
}
$env:PYTHONPATH = ''
$Wall | ConvertTo-Json | Set-Content -Encoding utf8 'cli-wall-seconds.json'
& $Python $Verdict cli $CliDir --output (Join-Path $CliDir 'cli-verdict.json')
Write-Output "Verdict cli: exit $LASTEXITCODE (0 = pass)"
Write-Output 'Gate runs complete.'
