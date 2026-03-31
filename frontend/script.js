// --- INITIALIZATION ---
lucide.createIcons();

const pdfZone = document.getElementById('drop-zone-pdf');
const pdfInput = document.getElementById('pdf-input');
const imgZone = document.getElementById('drop-zone-img');
const imgInput = document.getElementById('image-input');

// --- FILE BROWSER TRIGGERS ---
pdfZone.addEventListener('click', () => pdfInput.click());
imgZone.addEventListener('click', () => imgInput.click());

// Handle PDF Selection
pdfInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        document.getElementById('pdf-status').innerText = `File: ${e.target.files[0].name}`;
        pdfZone.classList.add('drop-zone-active');
    }
});

// Handle Image Selection
imgInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        document.getElementById('img-status').innerText = `Image: ${e.target.files[0].name}`;
        imgZone.classList.add('drop-zone-active');
        
        // Show immediate preview of original image
        const reader = new FileReader();
        reader.onload = (event) => {
            const resImg = document.getElementById('result-image');
            resImg.src = event.target.result;
            resImg.classList.remove('hidden');
            document.getElementById('vision-placeholder').classList.add('hidden');
        };
        reader.readAsDataURL(e.target.files[0]);
    }
});

// --- API ACTIONS ---

const showLoader = (show) => {
    document.getElementById('vision-loader').classList.toggle('hidden', !show);
};

const appendChat = (role, text, img = null) => {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = role === 'user' 
        ? "bg-blue-600 ml-auto p-3 rounded-xl text-sm max-w-[85%] text-right" 
        : "bg-slate-800 p-3 rounded-xl text-sm border border-slate-700 max-w-[85%]";
    
    let html = `<p>${text}</p>`;
    if (img) html += `<img src="${img}" class="mt-2 rounded border border-slate-600 max-h-32">`;
    
    div.innerHTML = html;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
};

async function handlePdfUpload() {
    if (!pdfInput.files[0]) return alert("Please select a PDF first");
    
    const formData = new FormData();
    formData.append('file', pdfInput.files[0]);

    showLoader(true);
    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        appendChat('assistant', "✅ " + data.message);
    } catch (err) {
        appendChat('assistant', "❌ Error indexing PDF.");
    } finally {
        showLoader(false);
    }
}

async function runVision(type) {
    if (!imgInput.files[0]) return alert("Please select an MRI/CT image");

    const formData = new FormData();
    formData.append('file', imgInput.files[0]);

    showLoader(true);
    try {
        const endpoint = type === 'vgg' ? '/predict_vgg' : (type === 'yolo' ? '/detect' : '/segment');
        const res = await fetch(endpoint, { method: 'POST', body: formData });

        if (type === 'vgg') {
            const data = await res.json();
            appendChat('assistant', `**Classification:** ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`);
        } else {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            document.getElementById('result-image').src = url;
            appendChat('assistant', `Vision analysis (${type.toUpperCase()}) completed.`);
        }
    } catch (err) {
        alert("Vision analysis failed.");
    } finally {
        showLoader(false);
    }
}

async function handleAsk() {
    const input = document.getElementById('query-input');
    const q = input.value.trim();
    if (!q) return;

    appendChat('user', q);
    input.value = '';

    try {
        const res = await fetch(`/ask?q=${encodeURIComponent(q)}`);
        const data = await res.json();
        appendChat('assistant', data.answer, data.image);
    } catch (err) {
        appendChat('assistant', "No relevant context found in documents.");
    }
}

// --- VOICE LOGIC ---
const micBtn = document.getElementById('mic-btn');
let recorder;

micBtn.onclick = async () => {
    if (!recorder || recorder.state === "inactive") {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recorder = new MediaRecorder(stream);
        let chunks = [];
        recorder.ondataavailable = e => chunks.push(e.data);
        recorder.onstop = async () => {
            const blob = new Blob(chunks, { type: 'audio/webm' });
            const fd = new FormData();
            fd.append('file', blob, 'voice.webm');
            
            appendChat('assistant', "Transcribing...");
            const res = await fetch('/voice-query', { method: 'POST', body: fd });
            const data = await res.json();
            appendChat('assistant', data.answer, data.image);
        };
        recorder.start();
        micBtn.classList.add('text-red-500', 'animate-pulse');
    } else {
        recorder.stop();
        micBtn.classList.remove('text-red-500', 'animate-pulse');
    }
};