/* ==========================================================================
   Scriptura — frontend logic
   No frameworks: plain DOM + fetch. Everything the UI needs (upload,
   camera capture, recognition, history) lives here.
   ========================================================================== */

(function () {
  "use strict";

  const HISTORY_KEY = "scriptura-history";
  const THEME_KEY = "scriptura-theme";
  const MAX_HISTORY = 40;
  const MAX_CAPTURE_WIDTH = 640;

  // ---- Element references ----
  const el = (id) => document.getElementById(id);

  const themeToggle = el("themeToggle");
  const themeIcon = el("themeIcon");

  const tabUpload = el("tabUpload");
  const tabCamera = el("tabCamera");
  const paneUpload = el("paneUpload");
  const paneCamera = el("paneCamera");

  const dropzone = el("dropzone");
  const dropzoneEmpty = el("dropzoneEmpty");
  const previewImg = el("previewImg");
  const fileInput = el("fileInput");

  const cameraFeed = el("cameraFeed");
  const captureCanvas = el("captureCanvas");
  const captureCanvasPreview = el("captureCanvasPreview");
  const cameraIdle = el("cameraIdle");
  const btnStartCamera = el("btnStartCamera");
  const btnCapture = el("btnCapture");
  const btnRetake = el("btnRetake");

  const btnRecognize = el("btnRecognize");
  const btnClearInput = el("btnClearInput");

  const resultStage = el("resultStage");
  const resultPlaceholder = el("resultPlaceholder");
  const resultGhost = el("resultGhost");
  const resultText = el("resultText");
  const confidenceFill = el("confidenceFill");
  const confidenceValue = el("confidenceValue");
  const btnCopy = el("btnCopy");

  const historyList = el("historyList");
  const btnClearHistory = el("btnClearHistory");
  const toastStack = el("toastStack");
  const demoBanner = el("demoBanner");

  // ---- State ----
  let currentBlob = null;      // Blob to send to /api/predict
  let currentThumb = null;     // dataURL used for the history thumbnail
  let cameraStream = null;
  let lastRecognizedText = "";

  // ------------------------------------------------------------------
  // Theme
  // ------------------------------------------------------------------
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    themeIcon.className = theme === "dark" ? "bi bi-sun-fill" : "bi bi-moon-stars-fill";
    localStorage.setItem(THEME_KEY, theme);
  }

  (function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved) {
      applyTheme(saved);
    } else {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      applyTheme(prefersDark ? "dark" : "light");
    }
  })();

  themeToggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    applyTheme(current === "dark" ? "light" : "dark");
  });

  // ------------------------------------------------------------------
  // Tabs (Upload / Camera)
  // ------------------------------------------------------------------
  function switchTab(target) {
    const toUpload = target === "upload";
    tabUpload.classList.toggle("active", toUpload);
    tabCamera.classList.toggle("active", !toUpload);
    paneUpload.classList.toggle("d-none", !toUpload);
    paneCamera.classList.toggle("d-none", toUpload);
    if (toUpload) stopCamera();
  }
  tabUpload.addEventListener("click", () => switchTab("upload"));
  tabCamera.addEventListener("click", () => switchTab("camera"));

  // ------------------------------------------------------------------
  // Upload + drag & drop
  // ------------------------------------------------------------------
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keypress", (e) => {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleSelectedFile(file);
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) handleSelectedFile(file);
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
      showToast("Please choose an image file.", "danger");
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      showToast("That image is over 8MB — try a smaller file.", "danger");
      return;
    }
    currentBlob = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      currentThumb = e.target.result;
      previewImg.src = currentThumb;
      previewImg.classList.remove("d-none");
      dropzoneEmpty.classList.add("d-none");
      btnRecognize.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // ------------------------------------------------------------------
  // Camera capture
  // ------------------------------------------------------------------
  async function startCamera() {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      cameraFeed.srcObject = cameraStream;
      cameraFeed.classList.remove("d-none");
      cameraIdle.classList.add("d-none");
      captureCanvasPreview.classList.add("d-none");
      btnStartCamera.classList.add("d-none");
      btnCapture.classList.remove("d-none");
      btnRetake.classList.add("d-none");
    } catch (err) {
      showToast("Couldn't access the camera — check permissions.", "danger");
    }
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      cameraStream = null;
    }
  }

  function captureFrame() {
    const videoW = cameraFeed.videoWidth || 640;
    const videoH = cameraFeed.videoHeight || 480;
    const scale = Math.min(1, MAX_CAPTURE_WIDTH / videoW);
    captureCanvas.width = videoW * scale;
    captureCanvas.height = videoH * scale;

    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(cameraFeed, 0, 0, captureCanvas.width, captureCanvas.height);

    captureCanvasPreview.src = captureCanvas.toDataURL("image/jpeg", 0.9);
    currentThumb = captureCanvasPreview.src;
    captureCanvasPreview.classList.remove("d-none");
    cameraFeed.classList.add("d-none");
    btnCapture.classList.add("d-none");
    btnRetake.classList.remove("d-none");

    captureCanvas.toBlob(
      (blob) => {
        currentBlob = blob;
        btnRecognize.disabled = false;
      },
      "image/jpeg",
      0.9
    );

    stopCamera();
  }

  btnStartCamera.addEventListener("click", startCamera);
  btnCapture.addEventListener("click", captureFrame);
  btnRetake.addEventListener("click", startCamera);

  // ------------------------------------------------------------------
  // Clear input
  // ------------------------------------------------------------------
  btnClearInput.addEventListener("click", () => {
    currentBlob = null;
    currentThumb = null;
    fileInput.value = "";
    previewImg.src = "";
    previewImg.classList.add("d-none");
    dropzoneEmpty.classList.remove("d-none");

    captureCanvasPreview.classList.add("d-none");
    cameraFeed.classList.add("d-none");
    cameraIdle.classList.remove("d-none");
    btnStartCamera.classList.remove("d-none");
    btnCapture.classList.add("d-none");
    btnRetake.classList.add("d-none");
    stopCamera();

    btnRecognize.disabled = true;
    resetResultStage();
  });

  // ------------------------------------------------------------------
  // Recognize
  // ------------------------------------------------------------------
  btnRecognize.addEventListener("click", async () => {
    if (!currentBlob) return;

    setRecognizing(true);
    resetResultStage();

    try {
      const formData = new FormData();
      const filename = currentBlob.name || "capture.jpg";
      formData.append("image", currentBlob, filename);

      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Recognition failed.");
      }

      if (data.demo_mode && demoBanner) demoBanner.classList.remove("d-none");

      revealResult(data.text, data.confidence);
      addHistoryEntry(data.text, data.confidence, data.preview_image || currentThumb);
    } catch (err) {
      showToast(err.message || "Something went wrong.", "danger");
      resultPlaceholder.textContent = "recognition failed — try again";
      resultPlaceholder.classList.remove("d-none");
    } finally {
      setRecognizing(false);
    }
  });

  function setRecognizing(isBusy) {
    btnRecognize.disabled = isBusy || !currentBlob;
    btnRecognize.innerHTML = isBusy
      ? '<span class="ink-spinner"></span> Reading…'
      : '<i class="bi bi-magic"></i> Recognize text';
  }

  function resetResultStage() {
    resultPlaceholder.classList.remove("d-none");
    resultPlaceholder.textContent = "your recognized text will appear here…";
    resultGhost.classList.remove("show");
    resultGhost.textContent = "";
    resultText.classList.add("d-none");
    resultText.innerHTML = "";
    confidenceFill.style.width = "0%";
    confidenceValue.textContent = "—";
    btnCopy.disabled = true;
  }

  function revealResult(text, confidence) {
    resultPlaceholder.classList.add("d-none");
    lastRecognizedText = text;

    // Signature moment: a handwritten "ghost" of the text flashes in
    // and dissolves, then the typed (monospace) transcription types
    // itself out — visualizing handwriting resolving into text.
    resultGhost.textContent = text;
    resultGhost.classList.add("show");

    window.setTimeout(() => {
      resultText.classList.remove("d-none");
      typewriter(resultText, text, () => {
        btnCopy.disabled = false;
      });
    }, 420);

    // Confidence bar animates in parallel with a slight delay so it
    // reads as "measuring" rather than instant.
    window.setTimeout(() => {
      const pct = Math.round(confidence * 100);
      confidenceFill.style.width = pct + "%";
      confidenceValue.textContent = pct + "%";
    }, 150);
  }

  function typewriter(container, text, onDone) {
    container.innerHTML = "";
    const caret = document.createElement("span");
    caret.className = "caret";
    let i = 0;

    function step() {
      if (i < text.length) {
        container.textContent = text.slice(0, i + 1);
        container.appendChild(caret);
        i += 1;
        window.setTimeout(step, 22);
      } else {
        caret.remove();
        if (onDone) onDone();
      }
    }
    step();
  }

  btnCopy.addEventListener("click", async () => {
    if (!lastRecognizedText) return;
    try {
      await navigator.clipboard.writeText(lastRecognizedText);
      showToast("Copied to clipboard.", "success");
    } catch {
      showToast("Couldn't copy — select and copy manually.", "danger");
    }
  });

  // ------------------------------------------------------------------
  // History (persisted in localStorage)
  // ------------------------------------------------------------------
  function loadHistory() {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch {
      return [];
    }
  }

  function saveHistory(items) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_HISTORY)));
  }

  function addHistoryEntry(text, confidence, thumb) {
    const items = loadHistory();
    items.unshift({
      id: Date.now(),
      text,
      confidence,
      thumb: thumb || null,
      timestamp: new Date().toISOString(),
    });
    saveHistory(items);
    renderHistory();
  }

  function renderHistory() {
    const items = loadHistory();
    historyList.innerHTML = "";

    if (items.length === 0) {
      historyList.innerHTML = '<div class="history-empty">No predictions yet — recognize your first sample above.</div>';
      return;
    }

    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "history-card";

      const thumbHtml = item.thumb
        ? `<img class="history-thumb" src="${item.thumb}" alt="Sample thumbnail">`
        : `<div class="history-thumb d-flex align-items-center justify-content-center"><i class="bi bi-image"></i></div>`;

      const pct = Math.round((item.confidence || 0) * 100);
      const when = formatRelativeTime(item.timestamp);

      card.innerHTML = `
        ${thumbHtml}
        <div class="flex-grow-1 min-width-0">
          <div class="history-text">${escapeHtml(item.text || "(empty)")}</div>
          <div class="history-meta">${when}</div>
        </div>
        <div class="history-conf">${pct}%</div>
      `;
      card.addEventListener("click", () => {
        revealResult(item.text, item.confidence);
      });
      historyList.appendChild(card);
    });
  }

  btnClearHistory.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
    showToast("History cleared.", "success");
  });

  function formatRelativeTime(iso) {
    const then = new Date(iso).getTime();
    const diffSec = Math.max(1, Math.round((Date.now() - then) / 1000));
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.round(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.round(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.round(diffHr / 24);
    return `${diffDay}d ago`;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ------------------------------------------------------------------
  // Toasts
  // ------------------------------------------------------------------
  function showToast(message, kind) {
    const toast = document.createElement("div");
    const color = kind === "danger" ? "var(--color-danger)" : "var(--color-success)";
    toast.style.cssText = `
      background: var(--color-bg-elevated);
      border: 1px solid var(--color-border);
      border-left: 4px solid ${color};
      border-radius: 10px;
      padding: 0.65rem 1rem;
      font-size: 0.88rem;
      box-shadow: var(--shadow-lift);
      color: var(--color-ink);
      max-width: 320px;
      animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;
    toastStack.appendChild(toast);
    window.setTimeout(() => {
      toast.style.transition = "opacity 0.3s ease, transform 0.3s ease";
      toast.style.opacity = "0";
      toast.style.transform = "translateX(12px)";
      window.setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  renderHistory();
})();
