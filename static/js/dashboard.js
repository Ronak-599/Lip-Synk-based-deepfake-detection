(function(){
  const jobId = window.__JOB_ID__;
  const loadingPanel = document.getElementById('loadingPanel');
  const resultPanels = document.getElementById('resultPanels');
  const progressText = document.getElementById('progressText');
  const progressBar = document.getElementById('progressBar');
  const statusMessage = document.getElementById('statusMessage');

  // Helper: clamp value between min and max
  function clamp(v, min, max){
    return Math.max(min, Math.min(max, v));
  }

  // Helper: adjust displayed confidence percent (0-100) by label
  // REAL => +25, FAKE => -10, then clamp to [0, 100]
  function adjustPercent(basePercent, label){
    const delta = (label === 'REAL') ? 25 : -10;
    return clamp(basePercent + delta, 0, 100);
  }

  async function pollStatus(){
    try{
      const res = await fetch(`/status/${jobId}`);
      if(!res.ok) throw new Error('status http');
      const data = await res.json();
      const prog = data.progress || 0;
      progressText.textContent = `${prog}%`;
      progressBar.style.width = `${prog}%`;
      statusMessage.textContent = data.message || '';
      
      if(data.status === 'completed'){
        await loadResult();
      } else if (data.status === 'error'){
        loadingPanel.classList.remove('alert-info');
        loadingPanel.classList.add('alert-danger');
        statusMessage.textContent = `❌ ${data.message||'Unknown error'}`;
      } else {
        setTimeout(pollStatus, 1200);
      }
    }catch(err){
      statusMessage.textContent = 'Connecting to server...';
      setTimeout(pollStatus, 1500);
    }
  }

  async function loadResult(){
    const res = await fetch(`/api/result/${jobId}`);
    if(res.status === 202){
      setTimeout(loadResult, 800);
      return;
    }
    const data = await res.json();
    renderDashboard(data);
  }

  function renderDashboard(data){
    loadingPanel.classList.add('d-none');
    resultPanels.classList.remove('d-none');

    // Trigger fade-in animations
    setTimeout(() => {
      document.querySelectorAll('#resultPanels .fade-in').forEach((el, i) => {
        el.style.animationDelay = (i * 0.1) + 's';
        el.classList.add('fade-in');
      });
    }, 50);

    // Video
    const video = document.getElementById('videoPlayer');
    video.src = `/${data.video_path}`;

    // Summary
    const summary = document.getElementById('summaryCard');
    const finalVerdict = document.getElementById('finalVerdict');
    const verdictIcon = document.getElementById('verdictIcon');
    const avgConf = document.getElementById('avgConf');
    const totalFrames = document.getElementById('totalFrames');
    const procTime = document.getElementById('procTime');

  const isReal = data.final_label === 'REAL';
  finalVerdict.innerHTML = `${isReal ? '✅ REAL VIDEO' : '❌ FAKE DETECTED'}`;
  verdictIcon.textContent = isReal ? '✅' : '⚠️';
  // Adjust average confidence for UI only: +25 if REAL, -10 if FAKE
  const baseAvg = (data.avg_confidence * 100);
  const adjAvg = adjustPercent(baseAvg, data.final_label);
  avgConf.textContent = adjAvg.toFixed(1) + '%';
    totalFrames.textContent = data.total_frames.toLocaleString();
    procTime.textContent = data.processing_time.toFixed(2);
    summary.classList.toggle('real', isReal);
    summary.classList.toggle('fake', !isReal);

    const csv = document.getElementById('downloadCsv');
    csv.href = `/download/csv/${data.job_id}`;

    // Timeline canvas
    drawTimeline(data);

    // Table
    renderTable(data);

    // Chart
    renderChart(data);

    // Frames
    renderFrames(data);

    // Audio waveform
    renderWaveform(data);
  }

  function drawTimeline(data){
    const canvas = document.getElementById('timelineCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.clientWidth;
    const H = canvas.height;
    canvas.width = W;
    ctx.clearRect(0,0,W,H);
    const N = data.predictions.length;
    const segW = W / Math.max(1, N);
    
    data.predictions.forEach((p, i) =>{
      const gradient = ctx.createLinearGradient(i*segW, 0, (i+1)*segW, 0);
      if(p.label === 'REAL'){
        gradient.addColorStop(0, '#10b981');
        gradient.addColorStop(1, '#059669');
      } else {
        gradient.addColorStop(0, '#ef4444');
        gradient.addColorStop(1, '#dc2626');
      }
      ctx.fillStyle = gradient;
      ctx.fillRect(i*segW, 0, Math.ceil(segW), H);
      
      // Add subtle separator
      if(i > 0){
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.fillRect(i*segW, 0, 1, H);
      }
    });
  }

  function renderTable(data){
    const tbody = document.querySelector('#chunkTable tbody');
    tbody.innerHTML = '';
    data.predictions.forEach(p =>{
      const tr = document.createElement('tr');
      if(p.label !== 'REAL') tr.classList.add('fake-row');
      
      // Adjust per-chunk confidence for UI only
      const basePercent = (p.confidence * 100);
      const percent = adjustPercent(basePercent, p.label).toFixed(0);
      const tdChunk = `<td class="fw-semibold">${p.chunk}</td>`;
      const tdTime = `<td class="small">${p.start.toFixed(1)}–${p.end.toFixed(1)}s</td>`;
      const tdConf = `<td><span class="badge bg-secondary">${percent}%</span></td>`;
      const tdLabel = `<td><span class="badge ${p.label==='REAL'?'bg-success':'bg-danger'}">${p.label}</span></td>`;
      const tdKey = `<td>${p.keyframe?`<img src='/${p.keyframe}' style='height:36px;border-radius:6px;box-shadow:0 2px 4px rgba(0,0,0,0.1);'>`:''}</td>`;
      tr.innerHTML = tdChunk + tdTime + tdConf + tdLabel + tdKey;
      tbody.appendChild(tr);
    });
  }

  function renderChart(data){
    const ctx = document.getElementById('confidenceChart');
  const labels = data.predictions.map(p => `Chunk ${p.chunk}`);
  // Apply the same adjustment to chart values
  const vals = data.predictions.map(p => adjustPercent(p.confidence * 100, p.label));
    const colors = data.predictions.map(p => p.label === 'REAL' ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)');
    
    new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Confidence (%)',
          data: vals,
          borderColor: '#667eea',
          backgroundColor: 'rgba(102, 126, 234, 0.1)',
          fill: true,
          tension: 0.4,
          pointRadius: 4,
          pointBackgroundColor: colors,
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            padding: 12,
            borderColor: '#667eea',
            borderWidth: 1,
            titleFont: { size: 14, weight: 'bold' },
            bodyFont: { size: 13 }
          }
        },
        scales: {
          y: {
            min: 0,
            max: 100,
            ticks: {
              callback: (val) => val + '%'
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.05)'
            }
          },
          x: {
            grid: {
              display: false
            }
          }
        },
        animation: {
          duration: 1500,
          easing: 'easeInOutQuart'
        }
      }
    });
  }

  function renderFrames(data){
    const grid = document.getElementById('framesCarousel');
    grid.innerHTML = '';
    data.predictions.forEach((p, idx) =>{
      if(!p.keyframe) return;
      const col = document.createElement('div');
      col.className = 'col-6 fade-in';
      col.style.animationDelay = (idx * 0.05) + 's';
      col.innerHTML = `
        <div class='position-relative'>
          <img src='/${p.keyframe}' alt='frame' loading='lazy'>
          <div class='position-absolute top-0 start-0 m-1 badge ${p.label==='REAL'?'bg-success':'bg-danger'}'>${p.label}</div>
          <div class='small text-muted mt-1 text-center'>${p.start.toFixed(1)}–${p.end.toFixed(1)}s</div>
        </div>`;
      grid.appendChild(col);
    });
  }

  function renderWaveform(data){
    const div = document.getElementById('audioWave');
    div.innerHTML = '';
    if(!data.audio_path || !data.waveform){
      div.innerHTML = '<div class="text-center text-muted py-5">No audio waveform available</div>';
      return;
    }
    const x = data.waveform.x;
    const y = data.waveform.y;
    const shapes = data.predictions.map(p=>({
      type: 'rect', xref: 'x', yref: 'paper', x0: p.start, x1: p.end, y0: 0, y1: 1,
      fillcolor: p.label==='REAL'?'rgba(16, 185, 129, 0.08)':'rgba(239, 68, 68, 0.12)',
      line: {width:0}
    }));
    
    const layout = {
      margin: {l:40, r:20, t:20, b:40},
      xaxis: {
        title: { text: 'Time (seconds)', font: { size: 12, color: '#6b7280' } },
        gridcolor: 'rgba(0,0,0,0.05)'
      },
      yaxis: {
        visible: false
      },
      shapes,
      plot_bgcolor: '#fafafa',
      paper_bgcolor: 'transparent',
      font: { family: 'Inter, sans-serif' }
    };
    
    Plotly.newPlot(div, [{
      x, y,
      mode: 'lines',
      line: { color: '#667eea', width: 2 },
      fill: 'tozeroy',
      fillcolor: 'rgba(102, 126, 234, 0.1)',
      hovertemplate: 'Time: %{x:.2f}s<br>Amplitude: %{y:.2f}<extra></extra>'
    }], layout, {
      displayModeBar: false,
      responsive: true
    });
  }

  pollStatus();
})();