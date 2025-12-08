param(
    [int]$Port = 8000
)

$backendProcess = Start-Process -FilePath "uv" -ArgumentList "run adk api_server app --port $Port --allow_origins='*'" -PassThru
Write-Host "Backend started with PID $($backendProcess.Id)"
Write-Host "Starting Frontend..."
try {
    npm --prefix frontend run dev
} finally {
    Write-Host "Stopping Backend..."
    Stop-Process -Id $backendProcess.Id -Force
}
