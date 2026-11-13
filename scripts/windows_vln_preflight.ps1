<# Read-only inventory; writes only the requested NEW report directory.
No installation, registry/policy changes, driver changes, WSL enablement, or simulation launch.
Run once for EACH relevant Python environment using -Python <python.exe>.
#>
param(
    [string]$Python = "python",
    [string[]]$IsaacRoot = @(),
    [string]$Output = ".\data\outputs\phase6c-windows-preflight"
)
$ErrorActionPreference = "Stop"
if (Test-Path $Output) { throw "Output already exists; choose a new report directory." }
$null = New-Item -ItemType Directory -Path $Output
$os = Get-CimInstance Win32_OperatingSystem
$report = [ordered]@{
    utc = (Get-Date).ToUniversalTime().ToString("o")
    os = $os.Caption; version = $os.Version; architecture = $os.OSArchitecture
    ram_total_bytes = [int64]$os.TotalVisibleMemorySize * 1024
    ram_free_bytes = [int64]$os.FreePhysicalMemory * 1024
    cpu = @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors)
    disks = @(Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | Select-Object DeviceID, Size, FreeSpace)
    display = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
    note = "Use nvidia-smi/PyTorch for VRAM; CIM AdapterRAM can overflow. No simulator or kernel test executed."
}
if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
    $report.wsl_list = (& wsl.exe --list --verbose 2>&1 | Out-String)
} else { $report.wsl_list = "wsl.exe unavailable" }
$roots = @($IsaacRoot) + @("C:\isaacsim", "C:\IsaacLab", "$env:USERPROFILE\IsaacLab")
foreach ($base in @("$env:LOCALAPPDATA\ov\pkg", "$env:USERPROFILE\isaacsim")) {
    if (Test-Path $base) {
        $roots += $base
        $roots += @(Get-ChildItem -Directory $base | Where-Object { $_.Name -match "isaac" } | ForEach-Object FullName)
    }
}
$report.searched_roots = @($roots | Select-Object -Unique)
$report | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $Output "windows.json")
$arguments = @((Join-Path $PSScriptRoot "pretrained_vln_preflight.py"), "--output", (Join-Path $Output "python.json"))
foreach ($root in $report.searched_roots) { $arguments += @("--isaac-root", $root) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Python inventory failed. Windows report retained; supply the intended python.exe path." }
Write-Output "Saved inventory to $Output. Simulator initialization/rendering and model inference remain untested."
