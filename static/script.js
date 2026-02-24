document.addEventListener("DOMContentLoaded", function () {

  console.log("script.js loaded ✅");

  let useWebcam = false;
  let lastPrediction = "";
  let spelledBuffer = "";
  let liveInterval = null;

  // Elements
  const uploadTab = document.getElementById("uploadTab");
  const liveTab = document.getElementById("liveTab");
  const uploadSection = document.getElementById("uploadSection");
  const liveSection = document.getElementById("liveSection");
  const dropZone = document.getElementById("dropZone");
  const mediaUpload = document.getElementById("mediaUpload");
  const preview = document.getElementById("preview");
  const webcam = document.getElementById("webcam");
  const predictBtn = document.getElementById("predictBtn");
  const loader = document.getElementById("loader");
  const resultBox = document.getElementById("result");
  const resultHindi = document.getElementById("resultHindi");
  const resultTamil = document.getElementById("resultTamil");
  const spelledOutput = document.getElementById("spelledOutput");
  const langSelect = document.getElementById("langSelect");

  // ---------- Tabs ----------
  uploadTab.onclick = () => {
    stopLive();
    useWebcam = false;
    uploadSection.classList.remove("hidden");
    liveSection.classList.add("hidden");
    uploadTab.classList.add("active");
    liveTab.classList.remove("active");
  };

  liveTab.onclick = () => {
    useWebcam = true;
    uploadSection.classList.add("hidden");
    liveSection.classList.remove("hidden");
    liveTab.classList.add("active");
    uploadTab.classList.remove("active");
    startWebcam();
  };

  // ---------- Upload ----------
  dropZone.onclick = () => mediaUpload.click();

  dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.style.background = "#e0f2fe";
  };

  dropZone.ondragleave = () => {
    dropZone.style.background = "#f9fafb";
  };

  dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.style.background = "#f9fafb";
    handleMedia(e.dataTransfer.files[0]);
  };

  mediaUpload.onchange = () => handleMedia(mediaUpload.files[0]);

  function handleMedia(file) {
    if (!file) return;
    preview.innerHTML = "";
    const url = URL.createObjectURL(file);

    if (file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.src = url;
      img.style.maxWidth = "100%";
      preview.appendChild(img);
    } else {
      const video = document.createElement("video");
      video.src = url;
      video.controls = true;
      video.style.maxWidth = "100%";
      preview.appendChild(video);
    }
  }

  // ---------- Webcam ----------
  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      webcam.srcObject = stream;
    } catch {
      alert("⚠️ Webcam access denied");
    }
  }

  function stopLive() {
    if (liveInterval) clearInterval(liveInterval);
    liveInterval = null;
  }

  // ---------- Predict ----------
  predictBtn.onclick = async () => {
    loader.classList.remove("hidden");
    lastPrediction = "";
    spelledBuffer = "";
    spelledOutput.innerText = "";

    if (useWebcam) {
      loader.classList.add("hidden");
      startLivePrediction();
      return;
    }

    const file = mediaUpload.files[0];
    if (!file) {
      alert("Upload a file first");
      loader.classList.add("hidden");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const endpoint = file.type.startsWith("image/")
      ? "/predict_image"
      : "/predict_video";

    try {
      const res = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();
      showResult(data.prediction);
    } catch {
      alert("Server error ❌");
    }

    loader.classList.add("hidden");
  };

  // ---------- Live ----------
  function startLivePrediction() {
    stopLive();

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    liveInterval = setInterval(async () => {
      if (!useWebcam) return stopLive();

      canvas.width = webcam.videoWidth;
      canvas.height = webcam.videoHeight;
      ctx.drawImage(webcam, 0, 0);

      const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg"));
      const fd = new FormData();
      fd.append("file", blob);

      try {
        const res = await fetch("/predict_image", { method: "POST", body: fd });
        const data = await res.json();
        showResult(data.prediction);
      } catch {
        console.log("Live prediction error");
      }

    }, 800); // slower → more stable
  }

  // ---------- Output ----------
  function showResult(text) {
    if (!text || text === lastPrediction) return;
    lastPrediction = text;

    resultBox.innerText = text;
    speakText(text);
    translateText(text);

    if (/^[a-z0-9]$/i.test(text)) {
      spelledBuffer += text;
      spelledOutput.innerText = "Spelling: " + spelledBuffer;
    }
  }

  function speakText(text) {
    speechSynthesis.cancel(); // stop previous voice
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = langSelect.value === "hi" ? "hi-IN"
               : langSelect.value === "ta" ? "ta-IN"
               : "en-US";
    speechSynthesis.speak(utter);
  }

  function translateText(text) {
    const map = {
      hello: { hi: "नमस्ते", ta: "வணக்கம்" },
      yes: { hi: "हाँ", ta: "ஆம்" },
      no: { hi: "नहीं", ta: "இல்லை" },

       hello: { hi: "नमस्ते", ta: "வணக்கம்" },
    yes: { hi: "हाँ", ta: "ஆம்" },
    no: { hi: "नहीं", ta: "இல்லை" },
    help: { hi: "मदद", ta: "உதவி" },
    thankyou: { hi: "धन्यवाद", ta: "நன்றி" },
    sorry: { hi: "माफ़ करना", ta: "மன்னிக்கவும்" },

    // Alphabets
    a: { hi: "ए", ta: "அ" },
    b: { hi: "बी", ta: "பி" },
    c: { hi: "सी", ta: "சி" },
    d: { hi: "डी", ta: "டி" },
    e: { hi: "ई", ta: "இ" },
    f: { hi: "एफ", ta: "எஃப்" },
    g: { hi: "जी", ta: "ஜி" },
    h: { hi: "एच", ta: "எச்" },
    i: { hi: "आई", ta: "ஐ" },
    j: { hi: "जे", ta: "ஜே" },
    k: { hi: "के", ta: "கே" },
    l: { hi: "एल", ta: "எல்" },
    m: { hi: "एम", ta: "எம்" },
    n: { hi: "एन", ta: "என்" },
    o: { hi: "ओ", ta: "ஓ" },
    p: { hi: "पी", ta: "பி" },
    q: { hi: "क्यू", ta: "க்யூ" },
    r: { hi: "आर", ta: "ஆர்" },
    s: { hi: "एस", ta: "எஸ்" },
    t: { hi: "टी", ta: "டி" },
    u: { hi: "यू", ta: "யூ" },
    v: { hi: "वी", ta: "வி" },
    w: { hi: "डब्ल्यू", ta: "டபிள்யூ" },
    x: { hi: "एक्स", ta: "எக்ஸ்" },
    y: { hi: "वाई", ta: "வை" },
    z: { hi: "जेड", ta: "ஸெட்" },

    // Numbers
    0: { hi: "शून्य", ta: "பூஜ்யம்" },
    1: { hi: "एक", ta: "ஒன்று" },
    2: { hi: "दो", ta: "இரண்டு" },
    3: { hi: "तीन", ta: "மூன்று" },
    4: { hi: "चार", ta: "நான்கு" },
    5: { hi: "पाँच", ta: "ஐந்து" },
    6: { hi: "छह", ta: "ஆறு" },
    7: { hi: "सात", ta: "ஏழு" },
    8: { hi: "आठ", ta: "எட்டு" },
    9: { hi: "नौ", ta: "ஒன்பது" },

    you: { hi: "आप", ta: "நீங்கள்" },
    work: { hi: "काम", ta: "வேலை" },
    warn: { hi: "चेतावनी", ta: "எச்சரிக்கை" },
    specific: { hi: "विशिष्ट", ta: "குறிப்பிட்ட" },
    skin: { hi: "त्वचा", ta: "தோல்" },
    pray: { hi: "प्रार्थना", ta: "பிரார்த்தனை" },
    pain: { hi: "दर्द", ta: "வலி" },
    from: { hi: "से", ta: "இருந்து" },
    doctor: { hi: "डॉक्टर", ta: "மருத்துவர்" },
    bad: { hi: "बुरा", ta: "கெட்ட" },
    assistance: { hi: "सहायता", ta: "உதவி" },
    agree: { hi: "सहमत", ta: "ஒப்புக்கொள்" },

    // Video Words
    accident: { hi: "दुर्घटना", ta: "விபத்து" },
    call: { hi: "कॉल", ta: "அழை" },
    help: { hi: "मदद", ta: "உதவி" },
    hot: { hi: "गरम", ta: "சூடு" },
    lose: { hi: "खोना", ta: "இழப்பு" },
    thief: { hi: "चोर", ta: "திருடன்" }

    };

    const t = map[text.toLowerCase()];
    resultHindi.innerText = t ? t.hi : "-";
    resultTamil.innerText = t ? t.ta : "-";
  }

});
