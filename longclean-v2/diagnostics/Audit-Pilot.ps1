param(
    [string]$Study = (Split-Path $PSScriptRoot -Parent),
    [string]$Output = (Join-Path $PSScriptRoot 'pilot-audit.json')
)

$ErrorActionPreference = 'Stop'
$frozen = Get-Content -Raw -Encoding utf8 (Join-Path $Study 'frozen.json') | ConvertFrom-Json -AsHashtable
$summary = Get-Content -Raw -Encoding utf8 (Join-Path $Study 'summary.json') | ConvertFrom-Json -AsHashtable
foreach ($name in $frozen.code.Keys) {
    if ((Get-FileHash (Join-Path $Study $name)).Hash -ne $frozen.code[$name]) {
        throw "Frozen source hash mismatch: $name"
    }
}

$archive = Join-Path $Study 'evidence.zip'
$archiveHash = (Get-FileHash $archive).Hash.ToLowerInvariant()
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    $first = [System.Collections.Generic.List[object]]::new()
    $same = [ordered]@{}
    foreach ($case in $summary.cases) {
        $tag = $case.case.tag
        $hashes = @{}
        foreach ($method in ('fixed', 'replan')) {
            $arm = $case.arms[$method]
            $entry = $zip.GetEntry("pilot/$tag/$method/actions.jsonl")
            $stream = $entry.Open()
            try { $hashes[$method] = (Get-FileHash -InputStream $stream).Hash }
            finally { $stream.Dispose() }

            $seen = @{}
            $reader = [IO.StreamReader]::new($entry.Open())
            try {
                while ($null -ne ($line = $reader.ReadLine())) {
                    $row = $line | ConvertFrom-Json -AsHashtable
                    foreach ($event in $row.transitions) {
                        if (!$event.in_basket -or $event.step -le $arm.base_checkpoint.step -or
                            $event.object -eq 'cream_cheese_1' -or $seen.ContainsKey($event.object)) { continue }
                        $seen[$event.object] = $true
                        $first.Add([ordered]@{
                            tag = $tag
                            method = $method
                            step = $event.step
                            placed_object = $event.object
                            assigned_target = $row.target
                            at_episode_end = $event.step -eq $arm.steps
                        })
                    }
                }
            } finally { $reader.Dispose() }
        }
        $same[$tag] = $hashes.fixed -eq $hashes.replan
    }
} finally { $zip.Dispose() }

$counts = foreach ($method in ('fixed', 'replan')) {
    $rows = @($first.Where({ $_.method -eq $method }))
    [ordered]@{
        method = $method
        first_placements_per_episode_and_object = $rows.Count
        placements_outside_assigned_target = @($rows.Where({ $_.placed_object -ne $_.assigned_target })).Count
        placements_at_episode_end = @($rows.Where({ $_.at_episode_end })).Count
    }
}
[ordered]@{
    evidence_sha256 = $archiveHash
    counts = @($counts)
    fixed_replan_action_logs_identical = $same
    first_placements = $first
} | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $Output
$counts | ForEach-Object { [pscustomobject]$_ } | Format-Table
