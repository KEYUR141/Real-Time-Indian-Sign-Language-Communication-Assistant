"use strict";

const state = {
  stream: null,
  capturing: false,
  frames: 0,
  frameTimer: null,
  transcript: [],
  uploadedUrl: null,
};

const elements = {
  cameraVideo: document.querySelector("#cameraVideo"),
  viewport: document.querySelector("#viewport"),
  cameraState: document.querySelector("#cameraState"),
  cameraButton: document.querySelector("#cameraButton"),
  captureButton: document.querySelector("#captureButton"),
  videoUpload: document.querySelector("#videoUpload"),
  recordingPill: document.querySelector("#recordingPill"),
  frameCount: document.querySelector("#frameCount"),
  handCoverage: document.querySelector("#handCoverage"),
  poseCoverage: document.querySelector("#poseCoverage"),
  emptyResult: document.querySelector("#emptyResult"),
  predictionResult: document.querySelector("#predictionResult"),
  recognizedSign: document.querySelector("#recognizedSign"),
  confidenceValue: document.querySelector("#confidenceValue"),
  rankedList: document.querySelector("#rankedList"),
  transcriptText: document.querySelector("#transcriptText"),
  latency: document.querySelector("#latency"),
  modeLabel: document.querySelector("#modeLabel"),
  clearButton: document.querySelector("#clearButton"),
  copyButton: document.querySelector("#copyButton"),
  speakButton: document.querySelector("#speakButton"),
  toast: document.querySelector("#toast"),
};

const samplePrediction = {
  label: "hat",
  confidence: 1,
  handCoverage: 0.813,
  poseCoverage: 1,
  latencyMs: 10.4,
  predictions: [
    ["hat", 1],
    ["old", 0],
    ["sick", 0],
    ["black", 0],
    ["india", 0],
  ],
};

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("show"), 2300);
}

function setPipeline(stageIndex) {
  document.querySelectorAll(".pipeline-list li").forEach((item, index) => {
    item.classList.toggle("complete", index < stageIndex);
    item.classList.toggle("active", index === stageIndex);
  });
}

async function startCamera() {
  if (state.stream) {
    stopCamera();
    return;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    elements.cameraVideo.srcObject = state.stream;
    elements.viewport.classList.add("has-video");
    elements.cameraState.classList.add("active");
    elements.cameraState.innerHTML = '<span class="dot"></span>Camera live';
    elements.cameraButton.innerHTML = '<span class="button-icon">■</span> Stop camera';
    elements.captureButton.disabled = false;
    setPipeline(0);
  } catch (error) {
    showToast("Camera permission was not granted. You can upload a video instead.");
  }
}

function stopCamera() {
  if (state.capturing) stopCapture(false);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  elements.cameraVideo.srcObject = null;
  elements.viewport.classList.remove("has-video");
  elements.cameraState.classList.remove("active");
  elements.cameraState.innerHTML = '<span class="dot"></span>Camera off';
  elements.cameraButton.innerHTML = '<span class="button-icon">◎</span> Start camera';
  elements.captureButton.disabled = true;
  setPipeline(-1);
}

function startCapture() {
  state.capturing = true;
  state.frames = 0;
  elements.frameCount.textContent = "0";
  elements.handCoverage.textContent = "--";
  elements.poseCoverage.textContent = "--";
  elements.recordingPill.classList.add("visible");
  elements.captureButton.classList.add("recording");
  elements.captureButton.innerHTML = '<span class="button-icon">■</span> Stop & classify';
  elements.emptyResult.hidden = false;
  elements.predictionResult.hidden = true;
  setPipeline(0);

  state.frameTimer = window.setInterval(() => {
    state.frames += 1;
    elements.frameCount.textContent = String(state.frames);
    const estimatedCoverage = Math.min(96, 48 + Math.round(state.frames * 1.25));
    elements.handCoverage.textContent = `${estimatedCoverage}%`;
    elements.poseCoverage.textContent = "100%";
  }, 110);
}

function stopCapture(classify = true) {
  state.capturing = false;
  window.clearInterval(state.frameTimer);
  elements.recordingPill.classList.remove("visible");
  elements.captureButton.classList.remove("recording");
  elements.captureButton.innerHTML = '<span class="button-icon">●</span> Start capture';

  if (!classify) return;
  if (state.frames < 16) {
    showToast("Capture at least 16 frames before classification.");
    setPipeline(0);
    return;
  }
  runPresentationPrediction();
}

function toggleCapture() {
  if (elements.captureButton.disabled) return;
  state.capturing ? stopCapture(true) : startCapture();
}

async function runPresentationPrediction() {
  elements.modeLabel.textContent = "Presentation sample";
  for (let stage = 1; stage <= 4; stage += 1) {
    setPipeline(stage);
    await new Promise((resolve) => window.setTimeout(resolve, 260));
  }
  await new Promise((resolve) => window.setTimeout(resolve, 240));
  setPipeline(5);
  showPrediction(samplePrediction);
}

function showPrediction(result) {
  elements.emptyResult.hidden = true;
  elements.predictionResult.hidden = false;
  elements.recognizedSign.textContent = result.label;
  elements.confidenceValue.textContent = `${Math.round(result.confidence * 100)}%`;
  elements.handCoverage.textContent = `${Math.round(result.handCoverage * 100)}%`;
  elements.poseCoverage.textContent = `${Math.round(result.poseCoverage * 100)}%`;
  elements.latency.textContent = `${result.latencyMs.toFixed(1)} ms`;
  elements.rankedList.innerHTML = result.predictions
    .map(([label, score], index) => {
      const percentage = Math.round(score * 100);
      return `<div class="rank-row">
        <span class="rank">${index + 1}</span>
        <div class="label" style="--score:${Math.max(percentage, 1)}%"><span>${label}</span></div>
        <span class="score">${percentage}%</span>
      </div>`;
    })
    .join("");
  state.transcript.push(result.label);
  renderTranscript();
}

function renderTranscript() {
  if (!state.transcript.length) {
    elements.transcriptText.innerHTML = "<span>Accepted signs will appear here.</span>";
    return;
  }
  elements.transcriptText.textContent = state.transcript.join(" ");
}

function resetCapture() {
  if (state.capturing) stopCapture(false);
  state.frames = 0;
  elements.frameCount.textContent = "0";
  elements.handCoverage.textContent = "--";
  elements.poseCoverage.textContent = "--";
  elements.latency.textContent = "-- ms";
  elements.emptyResult.hidden = false;
  elements.predictionResult.hidden = true;
  setPipeline(state.stream ? 0 : -1);
  showToast("Capture reset");
}

function loadVideo(event) {
  const [file] = event.target.files;
  if (!file) return;
  if (state.stream) stopCamera();
  if (state.uploadedUrl) URL.revokeObjectURL(state.uploadedUrl);
  state.uploadedUrl = URL.createObjectURL(file);
  elements.cameraVideo.srcObject = null;
  elements.cameraVideo.src = state.uploadedUrl;
  elements.cameraVideo.controls = true;
  elements.cameraVideo.loop = true;
  elements.cameraVideo.play();
  elements.viewport.classList.add("has-video");
  elements.cameraState.classList.add("active");
  elements.cameraState.innerHTML = '<span class="dot"></span>Video loaded';
  elements.captureButton.disabled = false;
  showToast(`${file.name} loaded`);
}

function clearTranscript() {
  state.transcript = [];
  renderTranscript();
  showToast("Recognized text cleared");
}

async function copyTranscript() {
  if (!state.transcript.length) return showToast("There is no recognized text to copy.");
  await navigator.clipboard.writeText(state.transcript.join(" "));
  showToast("Recognized text copied");
}

function speakTranscript() {
  if (!state.transcript.length) return showToast("There is no recognized text to speak.");
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(state.transcript.join(" ")));
}

elements.cameraButton.addEventListener("click", startCamera);
elements.captureButton.addEventListener("click", toggleCapture);
elements.videoUpload.addEventListener("change", loadVideo);
elements.clearButton.addEventListener("click", clearTranscript);
elements.copyButton.addEventListener("click", copyTranscript);
elements.speakButton.addEventListener("click", speakTranscript);

window.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !event.repeat) {
    event.preventDefault();
    toggleCapture();
  }
  if (event.key.toLowerCase() === "r" && !event.repeat) resetCapture();
});

window.addEventListener("beforeunload", () => {
  state.stream?.getTracks().forEach((track) => track.stop());
  if (state.uploadedUrl) URL.revokeObjectURL(state.uploadedUrl);
});
