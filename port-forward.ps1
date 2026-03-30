param([string]$RemoteHost = "172.25.186.200")

$ports = @(3000, 8000, 8002, 8003)

foreach ($port in $ports) {
    $job = Start-Job -ScriptBlock {
        param($lPort, $rHost, $rPort)
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $lPort)
        $listener.Start()
        Write-Output "Forwarding localhost:$lPort -> ${rHost}:${rPort}"
        while ($true) {
            $client = $listener.AcceptTcpClient()
            $remote = [System.Net.Sockets.TcpClient]::new($rHost, $rPort)
            $cs = $client.GetStream()
            $rs = $remote.GetStream()
            $job1 = [System.Threading.Tasks.Task]::Run([Action]{
                try { $cs.CopyTo($rs) } catch {} finally { $rs.Close() }
            })
            $job2 = [System.Threading.Tasks.Task]::Run([Action]{
                try { $rs.CopyTo($cs) } catch {} finally { $cs.Close() }
            })
            # Don't wait - let connections be handled in parallel
        }
    } -ArgumentList $port, $RemoteHost, $port
    Write-Host "Started forwarder for port $port (Job $($job.Id))"
}

Write-Host "`nPort forwarding active. Press Ctrl+C to stop."
Write-Host "Access services at:"
Write-Host "  Frontend:  http://localhost:3000"
Write-Host "  Gateway:   http://localhost:8000/docs"
Write-Host "  Stability: http://localhost:8002/docs"
Write-Host "  Viscosity: http://localhost:8003/docs"

try {
    while ($true) { Start-Sleep 60 }
} finally {
    Get-Job | Stop-Job -PassThru | Remove-Job
}
