param([string]$Study = $PSScriptRoot)

$ErrorActionPreference = 'Stop'
$freeze = Get-Content -Raw -Encoding utf8 (Join-Path $Study 'freeze.json') | ConvertFrom-Json -AsHashtable
$cases = @(
    foreach ($card in (2, 3)) {
        $report = Get-Content -Raw -Encoding utf8 (Join-Path $Study "blind-$card.json") | ConvertFrom-Json -AsHashtable
        if ($report.phase -ne 'blind') { throw 'Unexpected result phase' }
        foreach ($name in $freeze.source.Keys) {
            if ((Get-FileHash (Join-Path $Study $name)).Hash -ne $freeze.source[$name] -or
                $report.source[$name] -ne $freeze.source[$name]) { throw "Source hash differs for $name" }
        }
        $report.pairs
    }
)
$keys = @($cases | ForEach-Object { "$($_.layout)-$($_.condition)" })
$expected = @(foreach ($seed in $freeze.layouts) {
    "$seed-inside"
    if (($seed % 2) -eq 0) { "$seed-outside" }
})
if ($keys.Count -ne $expected.Count -or (Compare-Object $keys $expected)) { throw 'Unexpected case list' }
$unique = @{}
foreach ($p in $cases) {
    $a = $p.arms.vertical
    $b = $p.arms.lateral
    if ($a.initial.state_sha256 -ne $b.initial.state_sha256 -or
        $a.eligible -ne $b.eligible -or
        ($a.eligible -and ($a.checkpoint.state_sha256 -ne $b.checkpoint.state_sha256 -or
        ($a.checkpoint.observation_sha256 | ConvertTo-Json -Compress) -ne
        ($b.checkpoint.observation_sha256 | ConvertTo-Json -Compress)))) { throw "Unmatched pair $($p.layout)" }
    $hash = $a.initial.state_sha256
    if ($unique.ContainsKey($hash) -and $unique[$hash] -ne $p.layout) { throw "Repeated layout $($p.layout)" }
    $unique[$hash] = $p.layout
}
function Median($inputValues) {
    $v = @($inputValues | Sort-Object)
    if ($v.Count -eq 0) { return $null }
    ($v[[Math]::Floor(($v.Count-1)/2)] + $v[[Math]::Floor($v.Count/2)])/2
}
$inside = @($cases.Where({$_.condition -eq 'inside' -and $_.arms.vertical.eligible}))
$outside = @($cases.Where({$_.condition -eq 'outside' -and $_.arms.vertical.eligible}))
$gain = @($inside.Where({!$_.arms.vertical.visible_ink -and $_.arms.lateral.visible_ink})).Count
$loss = @($inside.Where({$_.arms.vertical.visible_ink -and !$_.arms.lateral.visible_ink})).Count
$arms = [ordered]@{}
foreach ($name in ('vertical','lateral')) {
    $rows = @($inside + $outside | ForEach-Object { $_.arms[$name] })
    $arms[$name] = [ordered]@{
        inside_visible_ink = @($inside.Where({$_.arms[$name].visible_ink})).Count
        outside_visible_ink = @($outside.Where({$_.arms[$name].visible_ink})).Count
        inside_median_peak_instance_pixels = Median ($inside | ForEach-Object { $_.arms[$name].max_visible_instance_pixels })
        inside_median_final_offset_mm = Median ($inside | ForEach-Object { 1000 * $_.arms[$name].actual_last_offset_m })
        unsafe_trials = @($rows.Where({!$_.safe})).Count
        max_tracked_shift_mm = 1000 * (($rows | ForEach-Object { $_.max_shift_m.Values } | Measure-Object -Maximum).Maximum)
    }
}
$gate = $inside.Count -ge 6 -and $outside.Count -ge 3 -and $gain -ge 3 -and $loss -eq 0 -and
    $arms.vertical.unsafe_trials -eq 0 -and $arms.lateral.unsafe_trials -eq 0
[ordered]@{
    inside_pairs = $inside.Count
    outside_pairs = $outside.Count
    gains = $gain
    losses = $loss
    arms = $arms
    gate = $gate
} | ConvertTo-Json -Depth 6 | Tee-Object -FilePath (Join-Path $Study 'summary.json')
