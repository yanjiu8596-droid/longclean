[CmdletBinding()]
param(
    [string]$SshHost = "connect.nmb1.seetacloud.com",
    [int]$Port = 23298,
    [string]$User = "root",
    [string]$IdentityFile,
    [string]$StatePath
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($IdentityFile)) {
    $IdentityFile = Join-Path $scriptRoot "..\..\.ssh_autodl\tableclear_ed25519_v2"
}
if ([string]::IsNullOrWhiteSpace($StatePath)) {
    $StatePath = Join-Path $scriptRoot "..\.test-artifacts\autodl-gpu-state.json"
}
$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$identity = [System.IO.Path]::GetFullPath($IdentityFile)
$stateFile = [System.IO.Path]::GetFullPath($StatePath)

if (-not (Test-Path -LiteralPath $ssh)) {
    throw "OpenSSH client is missing: $ssh"
}
if (-not (Test-Path -LiteralPath $identity)) {
    throw "SSH identity is missing: $identity"
}

$remoteScript = @'
set -u
gpu_present=0
gpu_name=""
gpu_memory=""
torch_cuda=0
torch_version=""

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    gpu_present=1
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
fi

if [ -x /root/miniconda3/bin/python ]; then
    torch_info=$(/root/miniconda3/bin/python - <<'PY' 2>/dev/null
import torch
print(f"{int(torch.cuda.is_available())}|{torch.__version__}")
PY
)
    if [ -n "${torch_info:-}" ]; then
        torch_cuda=${torch_info%%|*}
        torch_version=${torch_info#*|}
    fi
fi

printf 'GPU_PRESENT=%s\n' "$gpu_present"
printf 'GPU_NAME=%s\n' "$gpu_name"
printf 'GPU_MEMORY=%s\n' "$gpu_memory"
printf 'TORCH_CUDA=%s\n' "$torch_cuda"
printf 'TORCH_VERSION=%s\n' "$torch_version"
'@

$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$raw = @(
    & $ssh `
        -p $Port `
        -o BatchMode=yes `
        -o ConnectTimeout=12 `
        -o ServerAliveInterval=15 `
        -i $identity `
        "$User@$SshHost" `
        $remoteScript 2>&1
)
$sshExit = $LASTEXITCODE
$ErrorActionPreference = $previousErrorAction

$checkedAt = [DateTimeOffset]::UtcNow.ToString("o")
if ($sshExit -ne 0) {
    $result = [ordered]@{
        checked_at = $checkedAt
        instance_online = $false
        gpu_present = $false
        torch_cuda = $false
        ready_for_work = $false
        transition = "unreachable"
        detail = ($raw -join "`n")
    }
} else {
    $values = @{}
    foreach ($line in $raw) {
        if ($line -match "^([A-Z_]+)=(.*)$") {
            $values[$Matches[1]] = $Matches[2]
        }
    }
    $gpuPresent = $values["GPU_PRESENT"] -eq "1"
    $torchCuda = $values["TORCH_CUDA"] -eq "1"
    $ready = $gpuPresent -and $torchCuda
    $result = [ordered]@{
        checked_at = $checkedAt
        instance_online = $true
        gpu_present = $gpuPresent
        gpu_name = $values["GPU_NAME"]
        gpu_memory = $values["GPU_MEMORY"]
        torch_cuda = $torchCuda
        torch_version = $values["TORCH_VERSION"]
        ready_for_work = $ready
        transition = "unchanged"
    }
}

$previousReady = $false
if (Test-Path -LiteralPath $stateFile) {
    try {
        $previous = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        $previousReady = [bool]$previous.ready_for_work
    } catch {
        $previousReady = $false
    }
}

if ($result.ready_for_work -and -not $previousReady) {
    $result.transition = "not_ready_to_ready"
} elseif (-not $result.ready_for_work -and $previousReady) {
    $result.transition = "ready_to_not_ready"
}

$stateDirectory = Split-Path -Parent $stateFile
New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
$result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $stateFile -Encoding UTF8
$result | ConvertTo-Json -Depth 4

if ($result.transition -eq "not_ready_to_ready") {
    exit 0
}
if (-not $result.instance_online) {
    exit 1
}
if (-not $result.gpu_present) {
    exit 2
}
if ($result.ready_for_work) {
    exit 3
}
exit 4
