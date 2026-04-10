import { useState, useRef, useCallback, useEffect } from 'react';
import {
  Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler,
  Tooltip, Legend, CategoryScale, LinearScale, BarElement, ArcElement,
} from 'chart.js';
import { Radar, Bar, Line } from 'react-chartjs-2';
import './App.css';

ChartJS.register(
  RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend,
  CategoryScale, LinearScale, BarElement, ArcElement
);

const API_URL = 'http://localhost:8000';

function fmt(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function fmtSize(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

const STEPS = [
  { label: 'Frame Extraction', desc: 'Sampling 16 uniformly-spaced frames' },
  { label: 'Preprocessing', desc: 'Resize 224x224, ImageNet normalization' },
  { label: 'ResNet-18', desc: 'Feature extraction via pretrained backbone' },
  { label: 'Classification', desc: 'Temporal pooling + FC + Softmax' },
];

const SHOT_INFO = {
  fast_left:  { type: 'Pace', arm: 'Left Arm', speed: '135-150 km/h', desc: 'Left-arm fast bowling with natural inswing to right-handed batters.' },
  fast_right: { type: 'Pace', arm: 'Right Arm', speed: '135-150 km/h', desc: 'Right-arm fast bowling with conventional swing and seam movement.' },
  leg_left:   { type: 'Spin', arm: 'Left Arm', speed: '70-90 km/h', desc: 'Left-arm wrist spin (chinaman) with sharp turn away from RHB.' },
  leg_right:  { type: 'Spin', arm: 'Right Arm', speed: '70-90 km/h', desc: 'Right-arm leg spin with leg break, googly, and flipper variations.' },
  off_left:   { type: 'Spin', arm: 'Left Arm', speed: '75-95 km/h', desc: 'Slow left-arm orthodox spin, turning into right-handed batters.' },
  off_right:  { type: 'Spin', arm: 'Right Arm', speed: '75-95 km/h', desc: 'Right-arm off spin with off break turning away from right-handers.' },
};

const CLASSES = ["fast_left", "fast_right", "leg_left", "leg_right", "off_left", "off_right"];

// Trajectory canvas
function TrajectoryCanvas({ trajectory }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!trajectory || !ref.current) return;
    const c = ref.current, ctx = c.getContext('2d'), W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const pX = W*0.2, pW = W*0.6, pY = H*0.05, pH = H*0.9;
    ctx.fillStyle = 'rgba(74,158,109,0.05)'; ctx.strokeStyle = 'rgba(74,158,109,0.15)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(pX, pY, pW, pH, 6); ctx.fill(); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.setLineDash([3,3]);
    [0.15,0.85].forEach(r => { const y=pY+pH*r; ctx.beginPath(); ctx.moveTo(pX+8,y); ctx.lineTo(pX+pW-8,y); ctx.stroke(); });
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(160,157,152,0.5)'; ctx.font = '10px "IBM Plex Mono",monospace'; ctx.textAlign = 'center';
    ctx.fillText('BOWLER END', W/2, pY+pH*0.08); ctx.fillText('BATTER END', W/2, pY+pH*0.97);
    if (trajectory.length < 2) return;
    const mX = x => pX+(x/100)*pW, mY = y => pY+pH*0.12+(y/100)*(pH*0.78);
    ctx.strokeStyle = '#4a9e6d'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(mX(trajectory[0].x), mY(trajectory[0].y));
    for (let i=1;i<trajectory.length;i++) ctx.lineTo(mX(trajectory[i].x), mY(trajectory[i].y));
    ctx.stroke();
    trajectory.forEach((p,i) => {
      const px=mX(p.x), py=mY(p.y), r = i===0?5:i===trajectory.length-1?5:2.5;
      ctx.beginPath(); ctx.arc(px,py,r,0,Math.PI*2);
      ctx.fillStyle = i===0?'#4a9e6d':i===trajectory.length-1?'#c7503a':'rgba(232,230,227,0.5)';
      ctx.fill();
    });
    const last = trajectory[trajectory.length-1];
    ctx.fillStyle = '#e8e6e3'; ctx.font = 'bold 10px "IBM Plex Mono",monospace'; ctx.textAlign = 'left';
    ctx.fillText(last.speed+' km/h', mX(last.x)+8, mY(last.y)+3);
  }, [trajectory]);
  return <canvas ref={ref} width={300} height={400} className="trajectory-canvas" />;
}

function ConfidenceRing({ value }) {
  const pct = Math.round(value * 100);
  return (
    <div className="confidence-ring" style={{'--pct': pct}}>
      <svg viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="52" className="ring-bg"/>
        <circle cx="60" cy="60" r="52" className="ring-fill"
          style={{ strokeDasharray: `${2*Math.PI*52}`, strokeDashoffset: `${2*Math.PI*52*(1-value)}` }}/>
      </svg>
      <div className="ring-text">
        <span className="ring-value">{pct}</span>
        <span className="ring-unit">%</span>
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [step, setStep] = useState(0);
  const [tab, setTab] = useState('overview');
  const inputRef = useRef(null);

  useEffect(() => {
    if (!loading) { setStep(0); return; }
    const id = setInterval(() => setStep(p => (p+1)%STEPS.length), 1800);
    return () => clearInterval(id);
  }, [loading]);

  const pick = useCallback(f => {
    const ext = '.'+f.name.split('.').pop().toLowerCase();
    if (!['.mp4','.avi','.mov','.mkv'].includes(ext)) { setError(`Unsupported: ${ext}`); return; }
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError(null);
  }, []);

  const clear = useCallback(() => {
    setFile(null); if(preview) URL.revokeObjectURL(preview);
    setPreview(null); setResult(null); setError(null);
    if(inputRef.current) inputRef.current.value='';
  }, [preview]);

  const run = useCallback(async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const fd = new FormData(); fd.append('file', file);
      const r = await fetch(`${API_URL}/predict`, {method:'POST', body:fd});
      if (!r.ok) { const e=await r.json().catch(()=>({})); throw new Error(e.detail||`Error ${r.status}`); }
      setResult(await r.json()); setTab('overview');
    } catch(e) { setError(e.message||'Connection failed'); }
    finally { setLoading(false); }
  }, [file]);

  const scores = result ? Object.entries(result.all_scores).sort((a,b)=>b[1]-a[1]) : [];
  const info = result ? SHOT_INFO[result.predicted_class] : null;

  const chartColors = { grid: 'rgba(255,255,255,0.04)', tick: '#706d68', font: 'DM Sans' };

  const radarData = result ? {
    labels: Object.keys(result.all_scores).map(fmt),
    datasets: [{ label:'Confidence', data: Object.values(result.all_scores).map(v=>v*100),
      backgroundColor:'rgba(74,158,109,0.12)', borderColor:'rgba(74,158,109,0.7)',
      borderWidth:2, pointBackgroundColor:'#4a9e6d', pointBorderColor:'#232529', pointRadius:3 }],
  } : null;

  const radarOpts = { responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}},
    scales:{r:{beginAtZero:true, max:100, ticks:{display:false,stepSize:25},
      grid:{color:chartColors.grid}, angleLines:{color:chartColors.grid},
      pointLabels:{color:'#a09d98', font:{size:10, family:chartColors.font}}}} };

  const barData = result ? {
    labels: scores.map(([n])=>fmt(n)),
    datasets: [{ data: scores.map(([,v])=>v*100),
      backgroundColor: scores.map(([n])=> n===result.predicted_class ? 'rgba(74,158,109,0.7)' : 'rgba(160,157,152,0.15)'),
      borderColor: scores.map(([n])=> n===result.predicted_class ? '#4a9e6d' : 'rgba(160,157,152,0.3)'),
      borderWidth:1, borderRadius:3, barPercentage:0.65 }],
  } : null;

  const barOpts = { indexAxis:'y', responsive:true, maintainAspectRatio:false,
    plugins:{legend:{display:false}, tooltip:{backgroundColor:'#232529', titleFont:{family:chartColors.font}, bodyFont:{family:chartColors.font}, callbacks:{label:c=>c.parsed.x.toFixed(1)+'%'}}},
    scales:{ x:{max:100, grid:{color:chartColors.grid}, ticks:{color:chartColors.tick, font:{size:10}, callback:v=>v+'%'}},
      y:{grid:{display:false}, ticks:{color:'#a09d98', font:{size:11, family:chartColors.font}}}} };

  const accData = result?.training_history ? {
    labels: result.training_history.epochs.map(e=>''+e),
    datasets: [
      { label:'Train Acc', data:result.training_history.train_acc, borderColor:'#4a9e6d', backgroundColor:'rgba(74,158,109,0.08)', fill:true, tension:0.35, pointRadius:2.5 },
      { label:'Val Acc', data:result.training_history.val_acc, borderColor:'#c49032', backgroundColor:'rgba(196,144,50,0.08)', fill:true, tension:0.35, pointRadius:2.5 },
    ],
  } : null;

  const lossData = result?.training_history ? {
    labels: result.training_history.epochs.map(e=>''+e),
    datasets: [
      { label:'Train Loss', data:result.training_history.train_loss, borderColor:'#c7503a', backgroundColor:'rgba(199,80,58,0.08)', fill:true, tension:0.35, pointRadius:2.5 },
      { label:'Val Loss', data:result.training_history.val_loss, borderColor:'#4a8fb5', backgroundColor:'rgba(74,143,181,0.08)', fill:true, tension:0.35, pointRadius:2.5 },
    ],
  } : null;

  const lineOpts = { responsive:true, maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#a09d98', font:{size:10,family:chartColors.font}, usePointStyle:true, pointStyle:'circle'}}, tooltip:{backgroundColor:'#232529'}},
    scales:{ x:{grid:{color:chartColors.grid}, ticks:{color:chartColors.tick, font:{size:9}}}, y:{grid:{color:chartColors.grid}, ticks:{color:chartColors.tick, font:{size:9}}}} };

  return (
    <div className="app-wrapper">
      <nav className="top-nav">
        <div className="nav-brand">
          <div className="nav-logo">CV</div>
          <div><div className="nav-title">CricVision</div><div className="nav-subtitle">Bowling Analysis System</div></div>
        </div>
        <div className="nav-badges">
          <span className="badge badge-green"><span className="badge-dot"/>LIVE</span>
          <span className="badge badge-purple">v2.0</span>
        </div>
      </nav>

      <main className="main-content">
        {!result && (
          <header className="hero">
            <div className="hero-badge">Deep Learning Powered</div>
            <h1>Cricket Bowling Action<br/>Classification System</h1>
            <p>Upload a bowling video clip. The system extracts 16 frames, runs them through a pre-trained ResNet-18 backbone, and classifies the bowling action across 6 categories.</p>
            <div className="hero-stats">
              <div className="hero-stat"><span className="hero-stat-value">90.1%</span><span className="hero-stat-label">Val Accuracy</span></div>
              <div className="hero-stat"><span className="hero-stat-value">2,573</span><span className="hero-stat-label">Videos</span></div>
              <div className="hero-stat"><span className="hero-stat-value">6</span><span className="hero-stat-label">Classes</span></div>
              <div className="hero-stat"><span className="hero-stat-value">ResNet-18</span><span className="hero-stat-label">Backbone</span></div>
            </div>
          </header>
        )}

        {!result && (
          <section className="upload-section">
            <div className="glass-card">
              <div className={`dropzone ${dragOver?'drag-over':''}`}
                onDrop={e=>{e.preventDefault();setDragOver(false);const f=e.dataTransfer.files[0];if(f)pick(f);}}
                onDragOver={e=>{e.preventDefault();setDragOver(true);}} onDragLeave={()=>setDragOver(false)}
                onClick={()=>inputRef.current?.click()}>
                <div className="dropzone-content">
                  <div className="dropzone-icon-wrapper">
                    <div className="dropzone-icon">
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                    </div>
                  </div>
                  <h3>Drop your bowling clip here</h3>
                  <p>or <span className="browse-link">browse files</span> — .mp4 .avi .mov .mkv</p>
                </div>
              </div>
              <input ref={inputRef} type="file" className="file-input-hidden" accept=".mp4,.avi,.mov,.mkv,video/*" onChange={e=>{const f=e.target.files[0];if(f)pick(f);}}/>
              {file && (
                <div className="file-info-card">
                  <div className="file-header">
                    <div className="file-meta">
                      <div className="file-icon">MP4</div>
                      <div className="file-details"><h4>{file.name}</h4><span>{fmtSize(file.size)}</span></div>
                    </div>
                    <button className="remove-btn" onClick={clear}>✕</button>
                  </div>
                  <div className="video-preview"><video src={preview} controls preload="metadata"/></div>
                  <button className="analyze-btn" onClick={run} disabled={loading}>
                    <span>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                      Run Analysis
                    </span>
                  </button>
                </div>
              )}
              {error && <div className="error-message">{error}</div>}
            </div>
          </section>
        )}

        {loading && (
          <div className="loading-overlay">
            <div className="pipeline-visualizer">
              <div className="pipeline-title">Processing Video</div>
              <div className="pipeline-steps">
                {STEPS.map((s,i) => (
                  <div key={i} className={`pipeline-step ${i===step?'active':''} ${i<step?'done':''}`}>
                    <div className="pipeline-step-icon">{i+1}</div>
                    <div><div className="pipeline-step-label">{s.label}</div><div className="pipeline-step-desc">{s.desc}</div></div>
                  </div>
                ))}
              </div>
              <div className="pipeline-progress"><div className="pipeline-progress-fill" style={{width:`${((step+1)/STEPS.length)*100}%`}}/></div>
            </div>
          </div>
        )}

        {result && (
          <div className="results-dashboard">
            <div className="dashboard-header">
              <div className="dashboard-header-left">
                <button className="back-btn" onClick={clear}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="15 18 9 12 15 6"/></svg>
                  Back
                </button>
                <h2 className="dashboard-title">Analysis Results</h2>
              </div>
              <div className="dashboard-meta">
                <span className="meta-chip">{result.inference_time_seconds}s</span>
                <span className="meta-chip">{result.video_meta?.total_frames} frames</span>
              </div>
            </div>

            <div className="tab-nav">
              {[{id:'overview',l:'Overview'},{id:'trajectory',l:'Trajectory'},{id:'frames',l:'Frames'},{id:'training',l:'Metrics'}].map(t=>(
                <button key={t.id} className={`tab-btn ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.l}</button>
              ))}
            </div>

            {tab==='overview' && (
              <div className="tab-content overview-grid">
                <div className="glass-card result-hero-card">
                  <div className="result-hero-top">
                    <div className="result-hero-info">
                      <div className="result-label">Predicted Action</div>
                      <div className="result-class-name">{fmt(result.predicted_class)}</div>
                      <div className="result-tags">
                        <span className="result-tag tag-type">{info?.type}</span>
                        <span className="result-tag tag-arm">{info?.arm}</span>
                        <span className="result-tag tag-speed">{info?.speed}</span>
                      </div>
                      <p className="result-desc">{info?.desc}</p>
                    </div>
                    <ConfidenceRing value={result.confidence}/>
                  </div>
                </div>
                <div className="glass-card chart-card">
                  <div className="card-header"><div className="card-icon">R</div><div><h3>Confidence Radar</h3><p>6-class probability distribution</p></div></div>
                  <div className="chart-container radar-chart-container">{radarData && <Radar data={radarData} options={radarOpts}/>}</div>
                </div>
                <div className="glass-card chart-card">
                  <div className="card-header"><div className="card-icon">B</div><div><h3>Class Scores</h3><p>Softmax probabilities</p></div></div>
                  <div className="chart-container bar-chart-container">{barData && <Bar data={barData} options={barOpts}/>}</div>
                </div>
                <div className="glass-card meta-card" style={{gridColumn:'1/-1'}}>
                  <div className="card-header"><div className="card-icon">V</div><div><h3>Video Properties</h3><p>Input metadata</p></div></div>
                  <div className="meta-grid">
                    <div className="meta-item"><span className="meta-label">Resolution</span><span className="meta-value">{result.video_meta?.width}x{result.video_meta?.height}</span></div>
                    <div className="meta-item"><span className="meta-label">FPS</span><span className="meta-value">{result.video_meta?.fps}</span></div>
                    <div className="meta-item"><span className="meta-label">Frames</span><span className="meta-value">{result.video_meta?.total_frames}</span></div>
                    <div className="meta-item"><span className="meta-label">Duration</span><span className="meta-value">{result.video_meta?.duration}s</span></div>
                    <div className="meta-item"><span className="meta-label">Inference</span><span className="meta-value">{result.inference_time_seconds}s</span></div>
                    <div className="meta-item"><span className="meta-label">File Size</span><span className="meta-value">{file?fmtSize(file.size):'—'}</span></div>
                  </div>
                </div>
              </div>
            )}

            {tab==='trajectory' && (
              <div className="tab-content trajectory-layout">
                <div className="glass-card trajectory-card">
                  <div className="card-header"><div className="card-icon">T</div><div><h3>Ball Trajectory</h3><p>Simulated path — {fmt(result.predicted_class)}</p></div></div>
                  <div className="trajectory-wrapper">
                    <TrajectoryCanvas trajectory={result.trajectory}/>
                    <div className="trajectory-legend">
                      <div className="legend-item"><span className="legend-dot" style={{background:'#4a9e6d'}}/>Release</div>
                      <div className="legend-item"><span className="legend-dot" style={{background:'#c7503a'}}/>Impact</div>
                      <div className="legend-item"><span className="legend-dot" style={{background:'rgba(232,230,227,0.5)'}}/>Path</div>
                    </div>
                  </div>
                </div>
                <div className="glass-card trajectory-stats-card">
                  <div className="card-header"><div className="card-icon">M</div><div><h3>Trajectory Data</h3><p>Computed parameters</p></div></div>
                  <div className="trajectory-stats">
                    {result.trajectory && (()=>{
                      const pts=result.trajectory, avg=(pts.reduce((s,p)=>s+p.speed,0)/pts.length).toFixed(1),
                        lat=Math.max(...pts.map(p=>Math.abs(p.x-50))).toFixed(1);
                      return <>
                        <div className="stat-item"><span className="stat-label">Avg Speed</span><span className="stat-value">{avg} km/h</span></div>
                        <div className="stat-item"><span className="stat-label">Release Speed</span><span className="stat-value">{pts[0]?.speed} km/h</span></div>
                        <div className="stat-item"><span className="stat-label">Impact Speed</span><span className="stat-value">{pts[pts.length-1]?.speed} km/h</span></div>
                        <div className="stat-item"><span className="stat-label">Max Lateral Dev</span><span className="stat-value">{lat}%</span></div>
                        <div className="stat-item"><span className="stat-label">Points</span><span className="stat-value">{pts.length}</span></div>
                        <div className="stat-item"><span className="stat-label">Type</span><span className="stat-value">{info?.type}</span></div>
                      </>;
                    })()}
                  </div>
                </div>
              </div>
            )}

            {tab==='frames' && (
              <div className="tab-content">
                <div className="glass-card">
                  <div className="card-header"><div className="card-icon">F</div><div><h3>Extracted Keyframes</h3><p>{result.keyframes?.length} of {result.video_meta?.total_frames} frames sampled</p></div></div>
                  <div className="frames-grid">
                    {result.keyframes?.map((kf,i)=>(
                      <div key={i} className="frame-card">
                        <div className="frame-img-wrapper">
                          <img src={`data:image/jpeg;base64,${kf.data}`} alt={`Frame ${kf.frame_index}`}/>
                          <div className="frame-overlay"><span className="frame-badge">#{kf.frame_index}</span></div>
                        </div>
                        <div className="frame-info">
                          <span>Frame {kf.frame_index}</span>
                          <span className="frame-time">{result.video_meta?.fps>0?(kf.frame_index/result.video_meta.fps).toFixed(2)+'s':'—'}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="glass-card architecture-card">
                  <div className="card-header"><div className="card-icon">A</div><div><h3>Pipeline Architecture</h3><p>ResNet-18 video classification</p></div></div>
                  <div className="architecture-pipeline">
                    {[
                      {label:'Input',detail:'16 frames',color:'#4a9e6d'},
                      {label:'Normalize',detail:'224x224',color:'#4a8fb5'},
                      {label:'ResNet-18',detail:'512-d feat',color:'#c49032'},
                      {label:'Avg Pool',detail:'temporal',color:'#c49032'},
                      {label:'FC',detail:'512→6',color:'#c7503a'},
                      {label:'Softmax',detail:'6 probs',color:'#c7503a'},
                    ].map((s,i)=>(
                      <div key={i} className="arch-step">
                        <div className="arch-node" style={{borderColor:s.color}}><div className="arch-label">{s.label}</div><div className="arch-detail">{s.detail}</div></div>
                        {i<5 && <div className="arch-arrow">→</div>}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {tab==='training' && (
              <div className="tab-content training-grid">
                <div className="glass-card chart-card">
                  <div className="card-header"><div className="card-icon">A</div><div><h3>Accuracy</h3><p>Train vs Val over 15 epochs</p></div></div>
                  <div className="chart-container training-chart-container">
                    {accData && <Line data={accData} options={{...lineOpts, scales:{...lineOpts.scales, y:{...lineOpts.scales.y, min:30, max:100, ticks:{...lineOpts.scales.y.ticks, callback:v=>v+'%'}}}}}/>}
                  </div>
                </div>
                <div className="glass-card chart-card">
                  <div className="card-header"><div className="card-icon">L</div><div><h3>Loss</h3><p>CrossEntropy + label smoothing 0.1</p></div></div>
                  <div className="chart-container training-chart-container">{lossData && <Line data={lossData} options={lineOpts}/>}</div>
                </div>
                <div className="glass-card meta-card">
                  <div className="card-header"><div className="card-icon">C</div><div><h3>Config</h3><p>Training setup</p></div></div>
                  <div className="meta-grid">
                    <div className="meta-item"><span className="meta-label">Model</span><span className="meta-value">{result.model_info?.architecture}</span></div>
                    <div className="meta-item"><span className="meta-label">Input</span><span className="meta-value">{result.model_info?.input_shape}</span></div>
                    <div className="meta-item"><span className="meta-label">Optimizer</span><span className="meta-value">{result.model_info?.optimizer}</span></div>
                    <div className="meta-item"><span className="meta-label">Scheduler</span><span className="meta-value">{result.model_info?.scheduler}</span></div>
                    <div className="meta-item"><span className="meta-label">Dataset</span><span className="meta-value">{result.model_info?.dataset_size} videos</span></div>
                    <div className="meta-item"><span className="meta-label">Accuracy</span><span className="meta-value">{result.model_info?.val_accuracy}%</span></div>
                  </div>
                </div>
                <div className="glass-card meta-card">
                  <div className="card-header"><div className="card-icon">6</div><div><h3>Classes</h3><p>Bowling categories</p></div></div>
                  <div className="class-grid">
                    {CLASSES.map(cls => {
                      const si=SHOT_INFO[cls], sc=result.all_scores[cls];
                      return (
                        <div key={cls} className={`class-card ${cls===result.predicted_class?'is-predicted':''}`}>
                          <div className="class-name">{fmt(cls)}</div>
                          <div className="class-type">{si.type} · {si.arm}</div>
                          <div className="class-score">{(sc*100).toFixed(1)}%</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <footer className="footer">
          <div className="footer-content">
            <div className="footer-tech">
              <span className="tech-badge">PyTorch</span><span className="tech-badge">ResNet-18</span>
              <span className="tech-badge">FastAPI</span><span className="tech-badge">React</span><span className="tech-badge">Chart.js</span>
            </div>
            <p>Cricket Bowling Action Classification · Deep Learning Project · 2026</p>
          </div>
        </footer>
      </main>
    </div>
  );
}
