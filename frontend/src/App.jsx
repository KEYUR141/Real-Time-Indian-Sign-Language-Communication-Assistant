import { useEffect, useRef, useState } from 'react'
import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from '@mediapipe/tasks-vision'

const percent = (value) => `${(value * 100).toFixed(value > 0.995 ? 0 : 1)}%`
const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]

function browserFrame(hands, pose, timestampMs) {
  const coordinates = Array.from({ length: 51 }, () => [0, 0, 0])
  const confidence = Array(51).fill(0)
  const detected = Array(51).fill(false)
  const bestScore = { left: -1, right: -1 }

  hands.landmarks.forEach((points, index) => {
    const category = hands.handednesses?.[index]?.[0]
    const side = (category?.categoryName || category?.displayName || '').toLowerCase()
    const score = Math.max(0, Math.min(1, category?.score ?? 0))
    if (!(side in bestScore) || score <= bestScore[side]) return
    const offset = side === 'left' ? 0 : 21
    points.forEach((point, pointIndex) => {
      coordinates[offset + pointIndex] = [point.x, point.y, point.z]
      confidence[offset + pointIndex] = score
      detected[offset + pointIndex] = true
    })
    bestScore[side] = score
  })

  const posePoints = pose.landmarks?.[0]
  if (posePoints) {
    POSE_INDICES.forEach((sourceIndex, outputIndex) => {
      const point = posePoints[sourceIndex]
      const target = 42 + outputIndex
      coordinates[target] = [point.x, point.y, point.z]
      confidence[target] = Math.max(0, Math.min(1, point.visibility ?? 1))
      detected[target] = true
    })
  }
  return { coordinates, confidence, detected, timestamp_ms: timestampMs }
}

function StatusDot({ ok }) {
  return <span className={`status-dot ${ok ? 'online' : ''}`} aria-hidden="true" />
}

function App() {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const captureTimerRef = useRef(null)
  const canvasRef = useRef(null)
  const captureActiveRef = useRef(false)
  const landmarkFramesRef = useRef([])
  const captureStartedRef = useRef(0)
  const [health, setHealth] = useState(null)
  const [benchmark, setBenchmark] = useState(null)
  const [cameraOn, setCameraOn] = useState(false)
  const [recording, setRecording] = useState(false)
  const [phase, setPhase] = useState('idle')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [livePoints, setLivePoints] = useState(false)
  const [landmarkState, setLandmarkState] = useState('off')
  const [countdown, setCountdown] = useState(0)
  const [capturedFrames, setCapturedFrames] = useState(0)
  const [referenceSigns, setReferenceSigns] = useState([])
  const [referenceSign, setReferenceSign] = useState('hat')
  const [reference, setReference] = useState(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/health').then((response) => response.json()),
      fetch('/api/benchmark').then((response) => response.json()),
    ]).then(([status, metrics]) => {
      setHealth(status)
      setBenchmark(metrics)
    }).catch(() => setHealth({ status: 'offline', ready: false }))
    fetch('/api/reference/signs')
      .then((response) => {
        if (!response.ok) throw new Error('Reference library unavailable')
        return response.json()
      })
      .then((payload) => {
        setReferenceSigns(payload.signs)
        if (!payload.signs.includes('hat')) setReferenceSign(payload.signs[0] || '')
      })
      .catch(() => setReferenceSigns([]))
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop())
      window.clearTimeout(captureTimerRef.current)
    }
  }, [])

  useEffect(() => {
    if (!referenceSign) return
    fetch(`/api/reference/${encodeURIComponent(referenceSign)}`)
      .then((response) => {
        if (!response.ok) throw new Error('Reference video unavailable')
        return response.json()
      })
      .then(setReference)
      .catch(() => setReference(null))
  }, [referenceSign])

  useEffect(() => {
    if (!cameraOn || !livePoints) {
      const context = canvasRef.current?.getContext('2d')
      context?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      setLandmarkState('off')
      return undefined
    }

    let cancelled = false
    let animationFrame = 0
    let handLandmarker
    let poseLandmarker
    let lastVideoTime = -1
    setLandmarkState('loading')

    async function runLandmarks() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
        )
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/api/models/hand_landmarker.task', delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 2,
        })
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/api/models/pose_landmarker_lite.task', delegate: 'GPU' },
          runningMode: 'VIDEO',
          numPoses: 1,
        })
        if (cancelled) {
          handLandmarker.close()
          poseLandmarker.close()
          return
        }
        setLandmarkState('active')
        const canvas = canvasRef.current
        const context = canvas.getContext('2d')
        const drawing = new DrawingUtils(context)

        const render = () => {
          const video = videoRef.current
          if (!cancelled && video?.readyState >= 2) {
            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
              canvas.width = video.videoWidth
              canvas.height = video.videoHeight
            }
            if (video.currentTime !== lastVideoTime) {
              lastVideoTime = video.currentTime
              const timestamp = performance.now()
              const hands = handLandmarker.detectForVideo(video, timestamp)
              const pose = poseLandmarker.detectForVideo(video, timestamp)
              context.clearRect(0, 0, canvas.width, canvas.height)
              hands.landmarks.forEach((points) => {
                drawing.drawConnectors(points, HandLandmarker.HAND_CONNECTIONS, {
                  color: '#c8ff48', lineWidth: 3,
                })
                drawing.drawLandmarks(points, { color: '#f5ffe2', radius: 3 })
              })
              pose.landmarks.forEach((points) => {
                drawing.drawConnectors(points, PoseLandmarker.POSE_CONNECTIONS, {
                  color: '#75e7db', lineWidth: 2,
                })
                drawing.drawLandmarks(points, { color: '#bffbf5', radius: 2 })
              })
              if (captureActiveRef.current) {
                const elapsed = Math.max(
                  landmarkFramesRef.current.at(-1)?.timestamp_ms + 1 || 0,
                  Math.round(performance.now() - captureStartedRef.current),
                )
                landmarkFramesRef.current.push(browserFrame(hands, pose, elapsed))
                if (landmarkFramesRef.current.length % 5 === 0) {
                  setCapturedFrames(landmarkFramesRef.current.length)
                }
              }
            }
          }
          if (!cancelled) animationFrame = requestAnimationFrame(render)
        }
        render()
      } catch (landmarkError) {
        if (!cancelled) {
          setLandmarkState('error')
          setError(`Live points unavailable: ${landmarkError.message}`)
        }
      }
    }
    runLandmarks()
    return () => {
      cancelled = true
      cancelAnimationFrame(animationFrame)
      handLandmarker?.close()
      poseLandmarker?.close()
    }
  }, [cameraOn, livePoints])

  async function startCamera() {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      streamRef.current = stream
      videoRef.current.srcObject = stream
      setCameraOn(true)
      setLivePoints(true)
    } catch (cameraError) {
      setError(`Camera unavailable: ${cameraError.message}`)
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraOn(false)
    setLivePoints(false)
  }

  async function startRecording() {
    if (!streamRef.current || landmarkState !== 'active') return
    setResult(null)
    setError('')
    setPhase('countdown')
    for (let value = 3; value > 0; value -= 1) {
      setCountdown(value)
      await new Promise((resolve) => window.setTimeout(resolve, 700))
    }
    setCountdown(0)
    landmarkFramesRef.current = []
    captureStartedRef.current = performance.now()
    captureActiveRef.current = true
    setCapturedFrames(0)
    setResult(null)
    setRecording(true)
    setPhase('capture')
    captureTimerRef.current = window.setTimeout(() => {
      finishLandmarkCapture()
    }, 3200)
  }

  function stopRecording() {
    window.clearTimeout(captureTimerRef.current)
    finishLandmarkCapture()
  }

  function finishLandmarkCapture() {
    if (!captureActiveRef.current) return
    captureActiveRef.current = false
    setRecording(false)
    submitLandmarks([...landmarkFramesRef.current])
  }

  async function submitLandmarks(frames) {
    if (frames.length < 16) {
      setPhase('error')
      setError(`Only ${frames.length} landmark frames were captured. Keep your hands visible and try again.`)
      return
    }
    setError('')
    setResult(null)
    setPhase('processing')
    try {
      const response = await fetch('/api/predict/landmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Landmark prediction failed')
      setResult(payload)
      setPhase('done')
    } catch (requestError) {
      setPhase('error')
      setError(requestError.message)
    }
  }

  async function submitVideo(file, filename = file.name) {
    setError('')
    setResult(null)
    setPhase('processing')
    const form = new FormData()
    form.append('video', file, filename)
    try {
      const response = await fetch('/api/predict/video', { method: 'POST', body: form })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Prediction failed')
      setResult(payload)
      setPhase('done')
    } catch (requestError) {
      setPhase('error')
      setError(requestError.message)
    }
  }

  const backendOnline = health?.status === 'online' && health?.ready
  const stages = phase === 'done' ? 5 : phase === 'processing' ? 4 : phase === 'capture' ? 1 : 0

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Signal home">
          <span className="brand-symbol"><i /><i /><i /></span>
          <span>SIGNAL <small>/ ISL</small></span>
        </a>
        <div className="system-readout">
          <span><StatusDot ok={backendOnline} />{backendOnline ? 'SYSTEM ONLINE' : 'API OFFLINE'}</span>
          <span>{health?.device?.toUpperCase() || 'DEVICE —'}</span>
          <span>{health?.classes || 262} CLASSES</span>
        </div>
      </header>

      <main id="top">
        <section className="intro">
          <div>
            <p className="kicker">INDIAN SIGN LANGUAGE / LIVE INFERENCE</p>
            <h1>Motion, interpreted<br /><em>with intent.</em></h1>
          </div>
          <p className="intro-copy">One isolated sign. MediaPipe landmarks. A temporal Transformer. Readable language in a single, verifiable pipeline.</p>
        </section>

        <section className="reference-panel panel">
          <div className="panel-head"><span>REFERENCE LIBRARY / OFFICIAL INCLUDE TEST CLIPS</span><span>{referenceSigns.length || 0} SIGNS AVAILABLE</span></div>
          <div className="reference-layout">
            <div className="reference-copy">
              <p className="kicker">WATCH · PRACTISE · PERFORM</p>
              <h2>Learn the sign before you try it.</h2>
              <p>Select an official dataset example, observe the starting position and movement, then reproduce one complete sign in the live capture window.</p>
              <label className="sign-select">
                <span>REFERENCE SIGN</span>
                <select value={referenceSign} onChange={(event) => setReferenceSign(event.target.value)} disabled={!referenceSigns.length}>
                  {referenceSigns.map((sign) => <option key={sign} value={sign}>{sign.replaceAll('_', ' ')}</option>)}
                </select>
              </label>
              <div className="reference-actions">
                <button className="button primary" onClick={() => document.querySelector('.capture-card')?.scrollIntoView({ behavior: 'smooth' })}>Try this sign</button>
                <button className="button secondary" disabled={!reference?.file_path} onClick={() => navigator.clipboard.writeText(reference.file_path)}>Copy file path</button>
              </div>
              {reference && <p className="reference-source"><span>SOURCE</span>{reference.source}<small>{reference.filename}</small></p>}
            </div>
            <div className="reference-video">
              {reference ? <video key={reference.video_url} controls muted playsInline preload="metadata" src={reference.video_url}>Your browser cannot play this reference format.</video> : <div className="reference-empty"><strong>Reference library unavailable</strong><p>Restart the API to load official test videos.</p></div>}
              <div className="reference-tag"><span>EXPECTED SIGN</span><strong>{referenceSign || '—'}</strong></div>
            </div>
          </div>
        </section>

        <section className="demo-grid">
          <article className="capture-card panel">
            <div className="panel-head"><span>01 / INPUT</span><span>{recording ? '● RECORDING' : cameraOn ? 'CAMERA READY' : 'STANDBY'}</span></div>
            <div className={`viewport ${recording ? 'recording' : ''}`}>
              <video ref={videoRef} autoPlay muted playsInline />
              <canvas ref={canvasRef} className="landmark-canvas" aria-hidden="true" />
              {!cameraOn && <div className="empty-camera"><div className="orb"><span /></div><strong>Live sign capture</strong><p>Frame your upper body and keep both hands visible.</p></div>}
              <div className="reticle"><i /><i /><i /><i /></div>
              <div className="viewport-label"><span>MP / HANDS + POSE</span><span>{recording ? `${capturedFrames} FRAMES` : '64 TIMESTEPS'}</span></div>
              {livePoints && <div className={`points-state ${landmarkState}`}>{landmarkState === 'loading' ? 'LOADING VISION' : landmarkState === 'active' ? 'LIVE POINTS' : 'POINTS ERROR'}</div>}
              {countdown > 0 && <div className="capture-countdown"><span>GET READY</span><strong>{countdown}</strong><small>Begin when recording starts</small></div>}
            </div>
            <div className="controls">
              {!cameraOn ? <button className="button secondary" onClick={startCamera}>Enable camera</button> : <button className="button secondary" onClick={stopCamera} disabled={recording}>Disable camera</button>}
              <button className={`button primary ${recording ? 'stop' : ''}`} onClick={recording ? stopRecording : startRecording} disabled={!cameraOn || phase === 'processing' || countdown > 0 || landmarkState !== 'active'}>{recording ? 'Stop & interpret' : countdown > 0 ? `Starting in ${countdown}` : landmarkState === 'loading' ? 'Loading vision…' : 'Capture sign'}</button>
              <button className={`button points ${livePoints ? 'enabled' : ''}`} onClick={() => setLivePoints((enabled) => !enabled)} disabled={!cameraOn}>{livePoints ? 'Hide live points' : 'Show live points'}</button>
              <label className="button upload">Upload clip<input type="file" accept="video/*,.mov" onChange={(event) => event.target.files[0] && submitVideo(event.target.files[0])} /></label>
            </div>
            <p className="capture-note">A countdown gives you time to prepare. Perform one complete sign during the three-second window; landmarks go directly to the trained Transformer.</p>
          </article>

          <aside className="result-card panel">
            <div className="panel-head"><span>02 / DECISION</span><span>{phase.toUpperCase()}</span></div>
            {phase === 'processing' && <div className="waiting"><div className="loader" /><strong>Interpreting sequence</strong><p>Extracting landmarks and running temporal attention…</p></div>}
            {!result && phase !== 'processing' && <div className="waiting"><div className="signal-mark"><span /><span /><span /></div><strong>Awaiting a complete sign</strong><p>Capture or upload one isolated gesture.</p></div>}
            {result && <div className="result">
              <p className="micro">{result.accepted ? 'ACCEPTED PREDICTION' : 'UNKNOWN / REJECTED'}</p>
              <h2>{result.decision}</h2>
              <div className="confidence"><span>CONFIDENCE</span><strong>{percent(result.predictions[0].probability)}</strong></div>
              <div className="bar"><i style={{ width: percent(result.predictions[0].probability) }} /></div>
              <div className="coverage">
                <div><span>LEFT HAND</span><strong>{percent(result.quality.left_hand_coverage)}</strong></div>
                <div><span>RIGHT HAND</span><strong>{percent(result.quality.right_hand_coverage)}</strong></div>
                <div><span>POSE</span><strong>{percent(result.quality.pose_coverage)}</strong></div>
                <div><span>LATENCY</span><strong>{result.runtime.latency_ms} ms</strong></div>
              </div>
              <div className="rankings"><p>TOP CANDIDATES</p>{result.predictions.slice(1, 4).map((item) => <div key={item.label_id}><span>{item.rank.toString().padStart(2, '0')} / {item.label}</span><strong>{percent(item.probability)}</strong></div>)}</div>
            </div>}
            {error && <div className="error-box"><strong>Unable to interpret clip</strong><p>{error}</p></div>}
          </aside>
        </section>

        <section className="pipeline panel">
          <div className="panel-head"><span>LIVE PIPELINE</span><span>END-TO-END</span></div>
          <div className="flow">
            {[['01','VIDEO','camera / upload'],['02','LANDMARKS','51 points'],['03','SEQUENCE','64 × 255'],['04','ATTENTION','Transformer'],['05','DECISION','262 signs']].map((item, index) => <div className={index < stages ? 'active' : ''} key={item[0]}><b>{item[0]}</b><span>{item[1]}</span><small>{item[2]}</small></div>)}
          </div>
        </section>

        <section className="data-story panel" id="data-scale">
          <div className="panel-head"><span>DATA SCALE / VERIFIED INVENTORY</span><span>AI4BHARAT · ZENODO V1</span></div>
          <div className="data-layout">
            <div className="data-lead">
              <p className="kicker">FROM SOURCE VIDEO TO TRAINED SYSTEM</p>
              <h2><strong>{benchmark?.verified_videos?.toLocaleString() || '4,284'}</strong> verified ISL videos.</h2>
              <p>Collected from the official INCLUDE distribution by AI4Bharat and IIT Madras, then verified, converted into fixed landmark sequences, and split using the official manifests.</p>
              <div className="source-line"><span>SOURCE</span><strong>{benchmark?.provider || 'AI4Bharat / IIT Madras'}</strong><small>{benchmark?.distribution || 'Zenodo v1'}</small></div>
            </div>

            <div className="scale-grid">
              <div className="scale-metric hero-metric"><span>RAW VIDEO</span><strong>{benchmark?.extracted_video_gb || '53.84'} <small>GB</small></strong><p>Extracted and locally verified</p></div>
              <div className="scale-metric"><span>DOWNLOAD</span><strong>{benchmark?.compressed_download_gb || '56.8'} <small>GB</small></strong><p>{benchmark?.archive_count || 44} official archives</p></div>
              <div className="scale-metric"><span>VOCABULARY</span><strong>{benchmark?.classes || 262}</strong><p>Active sign classes</p></div>
              <div className="scale-metric"><span>CATEGORIES</span><strong>{benchmark?.semantic_categories || 15}</strong><p>Semantic groups</p></div>
            </div>
          </div>

          <div className="split-story">
            <div className="split-heading"><span>OFFICIAL MODEL SPLIT</span><small>4,284 MediaPipe sequences · 64 timesteps each</small></div>
            <div className="split-bar" aria-label="Dataset split proportions">
              <div className="train" style={{ flex: benchmark?.train_samples || 3121 }}><span>TRAIN</span><strong>{benchmark?.train_samples?.toLocaleString() || '3,121'}</strong></div>
              <div className="validation" style={{ flex: benchmark?.validation_samples || 347 }}><span>VALIDATION</span><strong>{benchmark?.validation_samples || 347}</strong></div>
              <div className="test" style={{ flex: benchmark?.test_samples || 816 }}><span>TEST</span><strong>{benchmark?.test_samples || 816}</strong></div>
            </div>
            <div className="impact-line"><span><b>99.81%</b> manifest coverage</span><span><b>{benchmark?.correct_samples || 770}/{benchmark?.test_samples || 816}</b> test videos correct</span><span><b>{benchmark ? percent(benchmark.test_macro_f1) : '94.47%'}</b> Macro F1</span></div>
          </div>
        </section>

        <section className="system-flow panel" id="how-it-works">
          <div className="panel-head"><span>HOW IT WORKS / SIMPLE VIEW</span><span>ONE SIGN · ONE RESULT</span></div>
          <div className="flow-intro">
            <div>
              <p className="kicker">FROM GESTURE TO READABLE TEXT</p>
              <h2>The system watches movement—not your identity.</h2>
            </div>
            <p>A normal camera is enough. No gloves or special sensors are required.</p>
          </div>

          <div className="simple-flow" aria-label="Sign recognition system flow">
            <div className="flow-step">
              <span className="step-number">01</span>
              <div className="step-icon camera-icon" aria-hidden="true"><i /></div>
              <h3>Perform a sign</h3>
              <p>The camera records one complete Indian Sign Language gesture.</p>
            </div>
            <span className="flow-arrow" aria-hidden="true">→</span>
            <div className="flow-step featured">
              <span className="step-number">02</span>
              <div className="step-icon nodes-icon" aria-hidden="true"><i /><i /><i /><i /><i /></div>
              <h3>Google MediaPipe</h3>
              <p>Google’s vision technology finds 51 important hand and upper-body points.</p>
              <small>21 + 21 HAND · 9 POSE</small>
            </div>
            <span className="flow-arrow" aria-hidden="true">→</span>
            <div className="flow-step">
              <span className="step-number">03</span>
              <div className="step-icon motion-icon" aria-hidden="true"><i /><i /><i /></div>
              <h3>Follow movement</h3>
              <p>The system tracks how those points move across 64 video frames.</p>
            </div>
            <span className="flow-arrow" aria-hidden="true">→</span>
            <div className="flow-step">
              <span className="step-number">04</span>
              <div className="step-icon brain-icon" aria-hidden="true"><i /><i /><i /></div>
              <h3>Transformer understands</h3>
              <p>The trained AI compares the movement pattern with 262 known signs.</p>
            </div>
            <span className="flow-arrow" aria-hidden="true">→</span>
            <div className="flow-step result-step">
              <span className="step-number">05</span>
              <div className="step-icon word-icon" aria-hidden="true">Aa</div>
              <h3>Show the word</h3>
              <p>A confident result becomes text; an uncertain sign becomes “Unknown”.</p>
            </div>
          </div>

          <div className="flow-summary">
            <span>CAMERA</span><i />
            <span>GOOGLE MEDIAPIPE</span><i />
            <span>MOVEMENT</span><i />
            <span>TRANSFORMER AI</span><i />
            <strong>TEXT</strong>
          </div>
        </section>
      </main>

      <footer>
        <span>INCLUDE / MEDIAPIPE / PYTORCH</span>
        <span>TEST ACCURACY <strong>{benchmark ? percent(benchmark.test_accuracy) : '94.36%'}</strong></span>
        <span>MACRO F1 <strong>{benchmark ? percent(benchmark.test_macro_f1) : '94.47%'}</strong></span>
      </footer>
    </div>
  )
}

export default App
