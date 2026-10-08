param(
    [string]$Server = '10.10.164.243',
    [int]$Port = 20339,
    [string]$RemoteProject = '/root/siton-data-efa478ddfa2a437485e4237f790c999b/ASTRA-Core-research-astra-tangent/ASTRA-Core-research-astra-tangent',
    [int]$Gpu = -1,
    [string]$PythonPath = '',
    [string]$ModelPath = '',
    [string]$DatasetRoot = ''
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskSshTarget = "root@$Server"
$taskScripts = @(
    (Join-Path $taskProjectRoot 'scripts/run_ofat_glue.py'),
    (Join-Path $taskProjectRoot 'scripts/start_ofat_glue_server.py')
)
$taskConfigs = @(
    (Join-Path $taskProjectRoot 'examples/sweep_ofat_cola_e15.json'),
    (Join-Path $taskProjectRoot 'examples/sweep_ofat_mrpc_e15.json'),
    (Join-Path $taskProjectRoot 'examples/sweep_ofat_sst2_e20.json')
)
foreach ($taskFile in @($taskScripts + $taskConfigs)) {
    if (-not (Test-Path -LiteralPath $taskFile -PathType Leaf)) {
        throw "Required file missing: $taskFile"
    }
}

# OpenSSH prompts for the password in this terminal; the script never stores it.
Write-Host 'Uploading training launchers. Enter the SSH password when prompted.'
& scp -P $Port @taskScripts "${taskSshTarget}:${RemoteProject}/scripts/"
if ($LASTEXITCODE -ne 0) { throw 'Launcher upload failed; training has not started.' }
Write-Host 'Uploading the three 34-run configurations.'
& scp -P $Port @taskConfigs "${taskSshTarget}:${RemoteProject}/examples/"
if ($LASTEXITCODE -ne 0) { throw 'Configuration upload failed; training has not started.' }

function ConvertTo-BashQuoted([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

$taskRemoteScript = ConvertTo-BashQuoted "$RemoteProject/scripts/start_ofat_glue_server.py"
$taskRemoteCommand = "python3 $taskRemoteScript --gpu $Gpu"
if ($PythonPath) { $taskRemoteCommand += ' --python-path ' + (ConvertTo-BashQuoted $PythonPath) }
if ($ModelPath) { $taskRemoteCommand += ' --model-path ' + (ConvertTo-BashQuoted $ModelPath) }
if ($DatasetRoot) { $taskRemoteCommand += ' --dataset-root ' + (ConvertTo-BashQuoted $DatasetRoot) }
Write-Host 'Checking the training environment and idle GPU, then starting the detached batch.'
& ssh -p $Port $taskSshTarget $taskRemoteCommand
if ($LASTEXITCODE -ne 0) { throw 'Server startup did not complete. Review the server error above.' }
