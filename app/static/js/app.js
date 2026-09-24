/**
 * Vision-Jev Guard Platform Front-End Controller
 * Real-time Multi-Camera hub, WebSocket telemetry, Jev decisions, SOP RAG & Webhook management.
 */

let ws = null;
let cameras = [];
let currentLayout = "2"; // 1 | 2 | auto
let presetList = [];     // [{id, name, ...}] loaded from /api/presets

/** Escapes a value for safe interpolation into HTML text or quoted attribute values. */
function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

/** Only allow inline image data URLs / same-origin paths as <img src>. */
function safeImageSrc(src) {
    const s = String(src || "");
    return /^data:image\/(png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=]+$/.test(s) || s.startsWith("/") ? s : "";
}

document.addEventListener("DOMContentLoaded", () => {
    initPresetSelector();
    initThresholdSelector();
    initLayoutButtons();
    initModals();
    initVisualRAGModal();
    initQuickTestButtons();
    initDelegatedActions();
    connectWebSocket();
    fetchPresets().then(fetchCameras);
    fetchWebhooks();
    fetchRAGDocs();
    fetchVisualRAGReferences();
});

function initThresholdSelector() {
    const sel = document.getElementById("select-threshold");
    if (!sel) return;
    sel.addEventListener("change", async (e) => {
        const val = parseFloat(e.target.value);
        try {
            await fetch("/api/settings/threshold", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ threshold: val })
            });
            console.log("Updated alert threshold to:", val);
        } catch (err) {
            console.error("Failed to update threshold:", err);
        }
    });
}

function initPresetSelector() {
    // Optional global preset selector if present
}

async function fetchPresets() {
    try {
        const resp = await fetch("/api/presets");
        presetList = await resp.json();
    } catch (e) {
        console.error("Failed to fetch presets:", e);
    }
}

function renderPresetOptions(selectedId) {
    const list = presetList.length ? presetList : [{ id: selectedId, name: selectedId }];
    return list.map(p =>
        `<option value="${escapeHtml(p.id)}" ${p.id === selectedId ? 'selected' : ''}>${escapeHtml(p.name)}</option>`
    ).join("");
}

// ===================== WebSocket & Telemetry =====================

function connectWebSocket() {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    const pulseEl = document.getElementById("ws-pulse");
    const statusTextEl = document.getElementById("ws-status-text");

    console.log("Connecting WebSocket:", wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("WebSocket connected.");
        if (pulseEl) pulseEl.className = "w-2 h-2 rounded-full bg-emerald-500 animate-pulse";
        if (statusTextEl) {
            statusTextEl.innerText = "ONLINE";
            statusTextEl.className = "text-[11px] text-emerald-400 font-semibold";
        }
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleIncomingMessage(data);
        } catch (err) {
            console.error("Failed to parse WebSocket message:", err);
        }
    };

    ws.onclose = () => {
        console.warn("WebSocket disconnected. Reconnecting in 2s...");
        if (pulseEl) pulseEl.className = "w-2 h-2 rounded-full bg-red-500";
        if (statusTextEl) {
            statusTextEl.innerText = "OFFLINE";
            statusTextEl.className = "text-[11px] text-red-400 font-semibold";
        }
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
    };
}

function handleIncomingMessage(data) {
    if (data.type === "initial_state") {
        if (data.cameras) renderCamerasGrid(data.cameras);
        if (data.metrics) updateMetricsHUD(data.metrics);
        return;
    }

    if (data.type === "cameras_updated") {
        if (data.cameras) renderCamerasGrid(data.cameras);
        return;
    }

    if (data.type === "decision_update") {
        hideDecisionError();
        renderDecisionTelemetry(data);
        if (data.metrics) updateMetricsHUD(data.metrics);
        return;
    }

    if (data.type === "decision_error") {
        showDecisionError(data.error, data.camera_name || data.camera_id, data.time_str);
    }
}

// 判定エンジン (推論サーバー) の失敗表示。CPU エミュレータへの自動切り替えはしない
function showDecisionError(error, cameraName, timeStr) {
    const container = document.getElementById("decision-cards-container");
    if (!container) return;
    let banner = document.getElementById("decision-error-banner");
    if (!banner) {
        banner = document.createElement("div");
        banner.id = "decision-error-banner";
        banner.className = "bg-red-950/70 border border-red-800/60 rounded-xl p-3 text-xs text-red-200 space-y-1";
        container.parentNode.insertBefore(banner, container);
    }
    const status = error && error.status ? ` (HTTP ${escapeHtml(error.status)})` : "";
    banner.innerHTML = `
        <div class="font-bold text-red-300">判定エンジンエラー${status}: ${escapeHtml(error ? error.kind : "unknown")}</div>
        <div class="font-mono break-all">${escapeHtml(error ? error.message : "")}</div>
        <div class="text-red-400/80">${escapeHtml(cameraName || "")} ${escapeHtml(timeStr || "")} — 判定結果は更新されていません</div>
    `;
    banner.classList.remove("hidden");
    container.classList.add("opacity-40");
}

function hideDecisionError() {
    const banner = document.getElementById("decision-error-banner");
    if (banner) banner.classList.add("hidden");
    const container = document.getElementById("decision-cards-container");
    if (container) container.classList.remove("opacity-40");
}

function updateMetricsHUD(metrics) {
    if (!metrics) return;

    // CPU
    if (metrics.system && metrics.system.cpu_percent !== undefined) {
        const cpuEl = document.getElementById("metric-cpu");
        const cpuBar = document.getElementById("metric-cpu-bar");
        const cpuVal = metrics.system.cpu_percent;
        if (cpuEl) cpuEl.innerText = `${cpuVal}%`;
        if (cpuBar) {
            cpuBar.style.width = `${Math.min(100, cpuVal)}%`;
            cpuBar.className = cpuVal > 80 ? "bg-red-500 h-1.5" : (cpuVal > 50 ? "bg-amber-500 h-1.5" : "bg-emerald-500 h-1.5");
        }
    }

    // Memory
    if (metrics.system && metrics.system.memory_used_mb) {
        const memEl = document.getElementById("metric-mem");
        if (memEl) memEl.innerText = `${metrics.system.memory_used_mb} MB (${metrics.system.memory_percent}%)`;
    }

    // WS Clients
    if (metrics.connections && metrics.connections.websocket_clients !== undefined) {
        const wsClientsEl = document.getElementById("metric-ws-clients");
        if (wsClientsEl) wsClientsEl.innerText = metrics.connections.websocket_clients;
    }

    // Latency
    if (metrics.inference && metrics.inference.total_latency_ms !== undefined) {
        const latEl = document.getElementById("metric-latency");
        if (latEl) latEl.innerText = `${metrics.inference.total_latency_ms} ms`;
    }
}

function renderDecisionTelemetry(data) {
    // 0. Scenario Override Freeze Banner
    const freezeBanner = document.getElementById("scenario-freeze-banner");
    const freezeSecVal = document.getElementById("freeze-countdown-val");
    if (data.is_scenario_override && data.remaining_override_sec > 0) {
        if (freezeBanner) freezeBanner.classList.remove("hidden");
        if (freezeSecVal) freezeSecVal.innerText = data.remaining_override_sec;
    } else {
        if (freezeBanner) freezeBanner.classList.add("hidden");
    }

    // 1. Focus Camera Tag
    const camTag = document.getElementById("focus-camera-tag");
    if (camTag) camTag.innerText = `CAM: ${(data.camera_name || data.camera_id || "PRIMARY").toUpperCase()}`;

    // 2. Vision State
    const stateEl = document.getElementById("vision-state-text");
    if (stateEl && data.state) stateEl.innerText = `"${data.state}"`;

    const timeEl = document.getElementById("vision-sample-time");
    if (timeEl && data.time_str) timeEl.innerText = data.time_str;

    // 2.5 DJev Diffusion Status Card
    if (data.decision && data.decision.djev) {
        const djev = data.decision.djev;
        const stepsEl = document.getElementById("djev-steps-val");
        const entropyEl = document.getElementById("djev-entropy-val");
        const multiEl = document.getElementById("djev-multimodal-val");
        const modelTag = document.getElementById("djev-model-tag");

        if (stepsEl) stepsEl.innerText = (djev.mode || "-").toUpperCase();
        if (entropyEl) {
            const usage = data.decision.usage || {};
            entropyEl.innerText = usage.input_tokens ? `${usage.input_tokens} tok` : "-";
        }
        if (multiEl) multiEl.innerText = djev.is_multimodal ? "MULTIMODAL" : "TEXT-ONLY";
        if (modelTag) modelTag.innerText = djev.model || "-";
    }

    // 2.7 Visual RAG Reference Card
    if (data.visual_rag) {
        updateVisualRAGCard(data.visual_rag);
    }

    // 3. Jev Decision Cards
    if (data.decision && data.decision.answers) {
        renderDecisionCards(data.decision.answers, data.decision.labels || {});
    }

    // 4. RAG SOP Action Card
    if (data.rag && data.rag.sop) {
        renderRAGSOP(data.rag.sop, data.rag.relevance_score);
    }

    // 5. Critical Alert Banner & Visual Pulse
    const isAlert = data.decision && data.decision.is_alert;
    const alertBanner = document.getElementById("alert-banner");
    const alertCamName = document.getElementById("alert-camera-name");
    const alertReason = document.getElementById("alert-reason");
    const alertTime = document.getElementById("alert-time");

    if (isAlert) {
        if (alertBanner) alertBanner.classList.remove("hidden");
        if (alertCamName) alertCamName.innerText = data.camera_name || data.camera_id;
        if (alertReason) alertReason.innerText = data.decision.alert_reason || "Threshold exceeded";
        if (alertTime) alertTime.innerText = data.time_str;

        // Highlight camera video container
        const targetCamBox = document.getElementById(`cam-box-${data.camera_id}`);
        if (targetCamBox) targetCamBox.classList.add("alert-active");
    } else {
        if (alertBanner) alertBanner.classList.add("hidden");
        const targetCamBox = document.getElementById(`cam-box-${data.camera_id}`);
        if (targetCamBox) targetCamBox.classList.remove("alert-active");
    }
}

// ===================== Decision Cards & SOP Renderers =====================

// answers は TypeSafe System One の Answer 形式 (choice / score / noul)
function renderDecisionCards(answers, labels) {
    const container = document.getElementById("decision-cards-container");
    if (!container) return;

    let html = "";
    for (const [qid, a] of Object.entries(answers)) {
        const label = labels[qid] || qid;
        if (a.type === "choice") {
            html += renderChoiceCard(a, label);
        } else if (a.type === "score") {
            html += renderScoreCard(a, label);
        } else if (a.type === "noul") {
            html += renderNoulCard(a, label);
        }
    }
    container.innerHTML = html;
}

function renderChoiceCard(q, label) {
    let choicesHtml = "";
    const sorted = Object.entries(q.probabilities || {}).sort((a, b) => b[1] - a[1]);

    for (const [choice, prob] of sorted) {
        const isSelected = choice === q.choice;
        const pct = Math.round(prob * 100);
        const barColor = isSelected ? (pct > 70 ? "bg-indigo-500" : "bg-blue-500") : "bg-slate-700";
        const textColor = isSelected ? "text-indigo-300 font-semibold" : "text-slate-400";

        choicesHtml += `
            <div class="space-y-1">
                <div class="flex justify-between text-xs">
                    <span class="${textColor} flex items-center gap-1.5">
                        ${isSelected ? '<span class="inline-block w-1.5 h-1.5 rounded-full bg-indigo-400"></span>' : ''}
                        ${escapeHtml(choice)}
                    </span>
                    <span class="text-slate-400 font-mono">${pct}%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div class="${barColor} h-1.5 rounded-full transition-all duration-300" style="width: ${pct}%"></div>
                </div>
            </div>
        `;
    }

    return `
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 shadow-sm">
            <div class="flex items-center justify-between mb-2.5">
                <div class="flex items-center gap-2">
                    <span class="px-2 py-0.5 text-[10px] font-bold rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 uppercase">CHOICE</span>
                    <h3 class="text-xs font-semibold text-slate-200">${escapeHtml(label)}</h3>
                </div>
                <span class="text-xs text-slate-400 font-mono">Conf: ${(q.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="space-y-2">
                ${choicesHtml}
            </div>
        </div>
    `;
}

function renderScoreCard(q, label) {
    // TypeSafe の score は段階番号の期待値 (0 〜 段階数-1)。色分けとバーは 0〜1 に正規化して表示
    const levels = Object.keys(q.legend || {}).length || 2;
    const scoreVal = (q.score || 0) / Math.max(1, levels - 1);
    const pct = Math.round(scoreVal * 100);
    const modal = Object.entries(q.probabilities || {}).sort((a, b) => b[1] - a[1])[0];
    const modalText = modal ? (q.legend || {})[modal[0]] : "";
    let colorClass = "text-emerald-400";
    let barColor = "bg-emerald-500";

    if (scoreVal >= 0.75) {
        colorClass = "text-red-400 font-bold";
        barColor = "bg-red-500";
    } else if (scoreVal >= 0.45) {
        colorClass = "text-amber-400 font-medium";
        barColor = "bg-amber-500";
    }

    return `
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 shadow-sm">
            <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                    <span class="px-2 py-0.5 text-[10px] font-bold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 uppercase">SCORE</span>
                    <h3 class="text-xs font-semibold text-slate-200">${escapeHtml(label)}</h3>
                </div>
                <span class="text-xs text-slate-400 font-mono">Conf: ${(q.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="flex items-end justify-between mb-1.5">
                <span class="text-2xl font-mono ${colorClass}">${(q.score || 0).toFixed(2)}</span>
                <span class="text-[11px] text-slate-400">Scale: 0 - ${levels - 1}</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden mb-1.5">
                <div class="${barColor} h-2 rounded-full transition-all duration-300" style="width: ${pct}%"></div>
            </div>
            ${modalText ? `<p class="text-[10px] text-slate-400 italic">${escapeHtml(modalText)}</p>` : ''}
        </div>
    `;
}

function renderNoulCard(q, label) {
    const isTrue = (q.noul || 0) >= 0.5;
    const badgeColor = isTrue
        ? "bg-red-500/20 text-red-400 border-red-500/40"
        : "bg-emerald-500/20 text-emerald-400 border-emerald-500/40";
    const statusText = isTrue ? "TRIGGERED (YES)" : "SAFE (NO)";
    const truePct = Math.round((q.noul || 0) * 100);

    return `
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 shadow-sm">
            <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                    <span class="px-2 py-0.5 text-[10px] font-bold rounded bg-purple-500/20 text-purple-400 border border-purple-500/30 uppercase">NOUL</span>
                    <h3 class="text-xs font-semibold text-slate-200">${escapeHtml(label)}</h3>
                </div>
                <span class="px-2 py-0.5 text-xs font-semibold rounded border ${badgeColor}">
                    ${statusText}
                </span>
            </div>
            <div class="space-y-1 mb-1.5">
                <div class="flex justify-between text-xs text-slate-400">
                    <span>P(yes)</span>
                    <span class="font-mono text-slate-200 font-semibold">${truePct}%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div class="${isTrue ? 'bg-red-500' : 'bg-emerald-500'} h-1.5 rounded-full transition-all duration-300" style="width: ${truePct}%"></div>
                </div>
            </div>
        </div>
    `;
}

function renderRAGSOP(sop, score) {
    if (!sop) return;

    const titleEl = document.getElementById("rag-sop-title");
    const summaryEl = document.getElementById("rag-sop-summary");
    const priorityEl = document.getElementById("rag-sop-priority");
    const stepsEl = document.getElementById("rag-sop-steps");
    const contactsEl = document.getElementById("rag-sop-contacts");

    if (titleEl) titleEl.innerText = sop.title || "Standard Operating Procedure";
    if (summaryEl) summaryEl.innerText = sop.summary || "";
    
    if (priorityEl) {
        priorityEl.innerText = sop.priority || "MEDIUM";
        priorityEl.className = sop.priority === "CRITICAL"
            ? "px-2 py-0.5 text-[10px] font-bold rounded bg-red-600 text-white animate-pulse"
            : (sop.priority === "HIGH" ? "px-2 py-0.5 text-[10px] font-bold rounded bg-amber-500 text-slate-950" : "px-2 py-0.5 text-[10px] font-bold rounded bg-slate-800 text-slate-300");
    }

    if (stepsEl && sop.procedure_steps) {
        stepsEl.innerHTML = sop.procedure_steps.map(s => `
            <div class="flex items-start gap-1.5">
                <span class="text-indigo-400 font-bold">&bull;</span>
                <span>${escapeHtml(s)}</span>
            </div>
        `).join("");
    }

    if (contactsEl && sop.emergency_contacts) {
        contactsEl.innerHTML = Object.entries(sop.emergency_contacts).map(([k, v]) => `
            <span class="px-2 py-1 rounded bg-slate-800/80 border border-slate-700 text-slate-300">
                <strong class="text-slate-400">${escapeHtml(k)}:</strong> ${escapeHtml(v)}
            </span>
        `).join("");
    }
}

// ===================== Multi-Camera Grid Management =====================

async function fetchCameras() {
    try {
        const resp = await fetch("/api/cameras");
        cameras = await resp.json();
        renderCamerasGrid(cameras);
    } catch (e) {
        console.error("Failed to fetch cameras:", e);
    }
}

function renderCamerasGrid(camList) {
    cameras = camList || [];
    const container = document.getElementById("cameras-grid");
    const badge = document.getElementById("active-cameras-badge");
    const hudCount = document.getElementById("metric-cameras-count");

    if (badge) badge.innerText = `${cameras.length} stream${cameras.length !== 1 ? 's' : ''}`;
    if (hudCount) hudCount.innerText = `${cameras.length} ACTIVE`;

    if (!container) return;

    if (cameras.length === 0) {
        container.innerHTML = `
            <div class="col-span-full bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center text-slate-500">
                アクティブな監視カメラがありません。「カメラ追加」ボタンからRTSPまたはJPEGストリームを登録してください。
            </div>
        `;
        return;
    }

    container.innerHTML = cameras.map(cam => {
        const camId = escapeHtml(cam.camera_id);
        const camIdPath = escapeHtml(encodeURIComponent(cam.camera_id));
        const statusColor = cam.status === "ACTIVE"
            ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
            : (cam.status.includes("FALLBACK") ? "text-amber-400 bg-amber-500/10 border-amber-500/30" : "text-red-400 bg-red-500/10 border-red-500/30");

        return `
            <div id="cam-box-${camId}" class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl flex flex-col transition-all duration-300 hover:border-slate-700">
                <!-- Camera Header -->
                <div class="px-3.5 py-2 bg-slate-850 border-b border-slate-800 flex items-center justify-between text-xs">
                    <div class="flex items-center gap-2 truncate">
                        <span class="w-2 h-2 rounded-full ${cam.status === 'ACTIVE' ? 'bg-emerald-500 animate-ping' : 'bg-amber-500'}"></span>
                        <span class="font-bold text-slate-200 truncate">${escapeHtml(cam.name)}</span>
                        <span class="text-[10px] font-mono text-slate-400">(${camId})</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="px-2 py-0.5 text-[10px] font-mono rounded border ${statusColor}">${escapeHtml(cam.status)}</span>
                        <button data-action="delete-camera" data-id="${camId}" class="text-slate-500 hover:text-red-400 p-1" title="カメラ削除">&times;</button>
                    </div>
                </div>

                <!-- Live Stream Video Frame -->
                <div class="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
                    <img src="/api/cameras/${camIdPath}/feed" alt="${escapeHtml(cam.name)}" class="w-full h-full object-contain">
                    <div class="absolute bottom-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-0.5 rounded text-[10px] font-mono text-slate-300">
                        ${escapeHtml(cam.actual_fps)} FPS &bull; ${escapeHtml(String(cam.source_type).toUpperCase())}
                    </div>
                </div>

                ${cam.error_message ? `
                <!-- Error / Fallback Notice Banner -->
                <div class="px-3 py-1.5 bg-amber-950/70 border-t border-b border-amber-800/40 text-[10px] text-amber-200 flex items-start gap-1.5">
                    <svg class="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                    <span class="leading-tight">${escapeHtml(cam.error_message)}</span>
                </div>
                ` : ''}

                <!-- Camera Controls & Preset Switcher -->
                <div class="p-3 bg-slate-900 flex items-center justify-between text-xs border-t border-slate-800/80">
                    <div class="flex items-center gap-1.5 text-slate-400">
                        <span>プリセット:</span>
                        <select data-action="update-camera-preset" data-id="${camId}" class="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200 text-xs focus:outline-none cursor-pointer">
                            ${renderPresetOptions(cam.preset_id)}
                        </select>
                    </div>
                    <span class="text-[11px] font-mono text-slate-500 truncate max-w-[140px]" title="${escapeHtml(cam.source_url)}">
                        ${escapeHtml(cam.source_url)}
                    </span>
                </div>
            </div>
        `;
    }).join("");
}

async function deleteCamera(camId) {
    if (!confirm(`カメラ '${camId}' を削除しますか？`)) return;
    try {
        await fetch(`/api/cameras/${encodeURIComponent(camId)}`, { method: "DELETE" });
        fetchCameras();
    } catch (e) {
        console.error("Failed to delete camera:", e);
    }
}

async function updateCameraPreset(camId, presetId) {
    try {
        await fetch(`/api/cameras/${encodeURIComponent(camId)}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ preset_id: presetId })
        });
    } catch (e) {
        console.error("Failed to update camera preset:", e);
    }
}

// Event delegation for controls rendered from server data (no inline handlers with interpolated ids)
function initDelegatedActions() {
    document.addEventListener("click", (e) => {
        const el = e.target.closest("[data-action]");
        if (!el || el.tagName === "SELECT") return;
        const id = el.dataset.id;
        switch (el.dataset.action) {
            case "delete-camera": deleteCamera(id); break;
            case "test-webhook": testWebhook(id); break;
            case "delete-webhook": deleteWebhook(id); break;
            case "delete-visual-reference": deleteVisualReference(id); break;
        }
    });
    document.addEventListener("change", (e) => {
        const el = e.target.closest("[data-action='update-camera-preset']");
        if (el) updateCameraPreset(el.dataset.id, el.value);
    });
}

// ===================== Modals & Actions =====================

function setSampleUrl(url, type, name, preset = null) {
    const urlInput = document.getElementById("add-cam-url");
    const typeSelect = document.getElementById("add-cam-type");
    const nameInput = document.getElementById("add-cam-name");
    const idInput = document.getElementById("add-cam-id");
    const fpsInput = document.getElementById("add-cam-fps");
    const presetSelect = document.getElementById("add-cam-preset");

    if (urlInput) urlInput.value = url;
    if (typeSelect) typeSelect.value = type;
    if (nameInput) nameInput.value = name;
    if (fpsInput) fpsInput.value = "1.0";
    if (preset && presetSelect) presetSelect.value = preset;
    if (idInput && !idInput.value) {
        const prefix = type === "hls" ? "hls_" : (type === "youtube" ? "yt_" : "cam_");
        idInput.value = prefix + Math.floor(Math.random() * 900 + 100);
    }
}

function initModals() {
    // Add Camera Modal
    const modalAddCam = document.getElementById("modal-add-cam");
    const btnOpenAddCam = document.getElementById("btn-open-add-cam");
    const btnCloseAddCam = document.getElementById("btn-close-add-cam");
    const btnCancelAddCam = document.getElementById("btn-cancel-add-cam");
    const formAddCam = document.getElementById("form-add-cam");

    if (btnOpenAddCam) btnOpenAddCam.addEventListener("click", () => modalAddCam.classList.remove("hidden"));
    if (btnCloseAddCam) btnCloseAddCam.addEventListener("click", () => modalAddCam.classList.add("hidden"));
    if (btnCancelAddCam) btnCancelAddCam.addEventListener("click", () => modalAddCam.classList.add("hidden"));

    if (formAddCam) {
        formAddCam.addEventListener("submit", async (e) => {
            e.preventDefault();
            const payload = {
                camera_id: document.getElementById("add-cam-id").value.trim(),
                name: document.getElementById("add-cam-name").value.trim(),
                source_type: document.getElementById("add-cam-type").value,
                source_url: document.getElementById("add-cam-url").value.trim(),
                preset_id: document.getElementById("add-cam-preset").value,
                sample_fps: parseFloat(document.getElementById("add-cam-fps").value) || 1.0
            };

            try {
                const resp = await fetch("/api/cameras", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const res = await resp.json();
                if (resp.ok) {
                    modalAddCam.classList.add("hidden");
                    formAddCam.reset();
                    fetchCameras();
                } else {
                    alert("カメラの追加に失敗しました: " + (res.detail || "エラー"));
                }
            } catch (err) {
                alert("通信エラー: " + err);
            }
        });
    }

    // Webhooks Modal
    const modalWebhooks = document.getElementById("modal-webhooks");
    const btnOpenWebhooks = document.getElementById("btn-open-webhooks");
    const btnCloseWebhooks = document.getElementById("btn-close-webhooks");
    const formAddWebhook = document.getElementById("form-add-webhook");

    if (btnOpenWebhooks) btnOpenWebhooks.addEventListener("click", () => {
        modalWebhooks.classList.remove("hidden");
        fetchWebhooks();
    });
    if (btnCloseWebhooks) btnCloseWebhooks.addEventListener("click", () => modalWebhooks.classList.add("hidden"));

    if (formAddWebhook) {
        formAddWebhook.addEventListener("submit", async (e) => {
            e.preventDefault();
            const payload = {
                id: "wh_" + Date.now(),
                name: document.getElementById("wh-name").value.trim(),
                format: document.getElementById("wh-format").value,
                url: document.getElementById("wh-url").value.trim(),
                min_score: parseFloat(document.getElementById("wh-min-score").value) || 0.75,
                cooldown_seconds: parseFloat(document.getElementById("wh-cooldown").value) || 30,
                enabled: true
            };

            try {
                const resp = await fetch("/api/webhooks", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (resp.ok) {
                    formAddWebhook.reset();
                    fetchWebhooks();
                }
            } catch (err) {
                console.error("Failed to save webhook:", err);
            }
        });
    }

    // RAG Modal
    const modalRAG = document.getElementById("modal-rag");
    const btnOpenRAG = document.getElementById("btn-open-rag");
    const btnCloseRAG = document.getElementById("btn-close-rag");

    if (btnOpenRAG) btnOpenRAG.addEventListener("click", () => {
        modalRAG.classList.remove("hidden");
        fetchRAGDocs();
    });
    if (btnCloseRAG) btnCloseRAG.addEventListener("click", () => modalRAG.classList.add("hidden"));
}

async function fetchWebhooks() {
    try {
        const resp = await fetch("/api/webhooks");
        const list = await resp.json();
        const container = document.getElementById("webhooks-list-container");
        if (!container) return;

        if (list.length === 0) {
            container.innerHTML = `<div class="text-slate-500 text-xs p-2">登録されているWebHookはありません。</div>`;
            return;
        }

        container.innerHTML = list.map(h => `
            <div class="bg-slate-950/70 border border-slate-800 rounded-lg p-3 flex items-center justify-between text-xs">
                <div>
                    <div class="flex items-center gap-2">
                        <strong class="text-slate-200">${escapeHtml(h.name)}</strong>
                        <span class="px-1.5 py-0.5 text-[10px] rounded bg-slate-800 text-slate-400 uppercase">${escapeHtml(h.format)}</span>
                    </div>
                    <div class="text-slate-500 font-mono text-[11px] truncate max-w-xs mt-0.5">${escapeHtml(h.url)}</div>
                </div>
                <div class="flex items-center gap-2">
                    <button data-action="test-webhook" data-id="${escapeHtml(h.id)}" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-emerald-400 font-semibold text-[11px]">テスト送信</button>
                    <button data-action="delete-webhook" data-id="${escapeHtml(h.id)}" class="text-slate-500 hover:text-red-400 text-base">&times;</button>
                </div>
            </div>
        `).join("");
    } catch (e) {
        console.error("Failed to fetch webhooks:", e);
    }
}

async function testWebhook(webhookId) {
    try {
        const resp = await fetch(`/api/webhooks/${encodeURIComponent(webhookId)}/test`, { method: "POST" });
        const res = await resp.json();
        if (res.success) {
            alert(`✅ テスト通知が正常に送信されました (HTTP ${res.status_code})`);
        } else {
            alert(`⚠️ 送信失敗: ${res.error || 'ステータスエラー ' + res.status_code}`);
        }
    } catch (e) {
        alert("エラー: " + e);
    }
}

async function deleteWebhook(webhookId) {
    if (!confirm("このWebHookを削除しますか？")) return;
    try {
        await fetch(`/api/webhooks/${encodeURIComponent(webhookId)}`, { method: "DELETE" });
        fetchWebhooks();
    } catch (e) {
        console.error("Failed to delete webhook:", e);
    }
}

async function fetchRAGDocs() {
    try {
        const resp = await fetch("/api/rag/documents");
        const list = await resp.json();
        const container = document.getElementById("rag-docs-list");
        if (!container) return;

        container.innerHTML = list.map(doc => `
            <div class="bg-slate-950/70 border border-slate-800 rounded-xl p-4 text-xs space-y-2">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2">
                        <span class="px-2 py-0.5 text-[10px] font-bold rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 uppercase">${escapeHtml(doc.category)}</span>
                        <strong class="text-sm text-slate-200">${escapeHtml(doc.title)}</strong>
                    </div>
                    <span class="px-2 py-0.5 text-[10px] font-bold rounded bg-slate-800 text-slate-400">${escapeHtml(doc.priority)}</span>
                </div>
                <p class="text-slate-400">${escapeHtml(doc.summary)}</p>
                <div class="bg-slate-900 rounded p-2.5 space-y-1 font-mono text-[11px] text-slate-300">
                    ${doc.procedure_steps.map(s => `<div>${escapeHtml(s)}</div>`).join("")}
                </div>
            </div>
        `).join("");
    } catch (e) {
        console.error("Failed to fetch RAG docs:", e);
    }
}

// ===================== Quick Test & Layouts =====================

function initQuickTestButtons() {
    const testBtns = document.querySelectorAll("[data-test-scenario]");
    testBtns.forEach(btn => {
        btn.addEventListener("click", async () => {
            const scenario = btn.getAttribute("data-test-scenario");
            const preset = btn.getAttribute("data-preset") || "security";

            // Immediately display freeze banner
            const freezeBanner = document.getElementById("scenario-freeze-banner");
            const freezeSecVal = document.getElementById("freeze-countdown-val");
            if (freezeBanner) freezeBanner.classList.remove("hidden");
            if (freezeSecVal) freezeSecVal.innerText = "20";

            try {
                const resp = await fetch("/api/trigger_scenario", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        state: scenario,
                        preset_id: preset,
                        camera_id: "cam_main",
                        freeze_seconds: 20
                    })
                });
                const res = await resp.json();
                if (!resp.ok) {
                    const err = (res.detail && res.detail.decision_engine_error) || { kind: "http", status: resp.status, message: JSON.stringify(res.detail) };
                    showDecisionError(err, "Manual Injection", new Date().toLocaleTimeString());
                    return;
                }
                hideDecisionError();
                console.log("Triggered test scenario result (freeze for 20s):", res);
                renderDecisionTelemetry({
                    type: "decision_update",
                    timestamp: Date.now() / 1000,
                    time_str: new Date().toLocaleTimeString(),
                    camera_id: "cam_main",
                    camera_name: "Primary Camera",
                    preset_id: preset,
                    state: scenario,
                    has_motion: true,
                    is_scenario_override: true,
                    remaining_override_sec: 20,
                    decision: {
                        latency_ms: res.latency_ms,
                        mode: res.mode,
                        answers: res.answers,
                        labels: res.labels,
                        usage: res.usage,
                        score: res.score,
                        is_alert: res.is_alert,
                        alert_reason: res.alert_reason,
                        alert_threshold: res.alert_threshold,
                        djev: res.djev || {}
                    },
                    rag: {
                        source: "embedded_rag",
                        matched: true,
                        relevance_score: 0.95,
                        latency_ms: 2.0,
                        sop: res.sop_action
                    }
                });
            } catch (e) {
                console.error("Failed to trigger test scenario:", e);
            }
        });
    });
}

async function resumeLiveMonitoring() {
    try {
        await fetch("/api/resume_live", { method: "POST" });
        const freezeBanner = document.getElementById("scenario-freeze-banner");
        if (freezeBanner) freezeBanner.classList.add("hidden");
        console.log("Resumed live surveillance.");
    } catch (e) {
        console.error("Failed to resume live surveillance:", e);
    }
}


function initLayoutButtons() {
    const layoutBtns = document.querySelectorAll("[data-layout]");
    const gridContainer = document.getElementById("cameras-grid");

    layoutBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            layoutBtns.forEach(b => b.className = "px-2 py-1 text-xs font-mono rounded text-slate-400 hover:text-white");
            btn.className = "px-2 py-1 text-xs font-mono rounded bg-slate-800 text-indigo-400 font-bold";

            const layout = btn.getAttribute("data-layout");
            currentLayout = layout;

            if (!gridContainer) return;
            if (layout === "1") {
                gridContainer.className = "grid grid-cols-1 gap-4";
            } else if (layout === "2") {
                gridContainer.className = "grid grid-cols-1 sm:grid-cols-2 gap-4";
            } else {
                gridContainer.className = "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4";
            }
        });
    });
}

// ===================== Visual Example RAG Controller =====================

function updateVisualRAGCard(visualRag) {
    if (!visualRag) return;

    const titleEl = document.getElementById("visual-rag-title");
    const simBadge = document.getElementById("visual-rag-sim-badge");
    const thumbImg = document.getElementById("visual-rag-thumb");
    const noThumb = document.getElementById("visual-rag-no-thumb");
    const anomScoreEl = document.getElementById("visual-rag-anomaly-score");
    const statusTag = document.getElementById("visual-rag-status-tag");

    const top = visualRag.top_match;
    const sim = visualRag.similarity || 0.0;
    const anom = visualRag.anomaly_score || 0.0;
    const isAnom = visualRag.is_anomalous;

    if (simBadge) {
        simBadge.innerText = `SIM: ${(sim * 100).toFixed(1)}%`;
        simBadge.className = isAnom
            ? "text-[10px] font-mono text-rose-300 bg-rose-950 px-2 py-0.5 rounded border border-rose-800"
            : "text-[10px] font-mono text-cyan-300 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800";
    }

    if (top) {
        if (titleEl) titleEl.innerText = top.title || "リファレンス照合中";
        if (thumbImg && top.image_base64) {
            thumbImg.src = safeImageSrc(top.image_base64);
            thumbImg.classList.remove("hidden");
            if (noThumb) noThumb.classList.add("hidden");
        }
    } else {
        if (titleEl) titleEl.innerText = "照合中...";
        if (thumbImg) thumbImg.classList.add("hidden");
        if (noThumb) noThumb.classList.remove("hidden");
    }

    if (anomScoreEl) {
        anomScoreEl.innerText = anom.toFixed(2);
        anomScoreEl.className = anom >= 0.70 ? "font-bold text-rose-400" : (anom >= 0.40 ? "font-bold text-amber-400" : "font-bold text-emerald-400");
    }

    if (statusTag) {
        if (isAnom) {
            statusTag.innerText = "⚠️ 過去異常一致";
            statusTag.className = "px-1.5 py-0.2 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30";
        } else {
            statusTag.innerText = "✅ 正常照合";
            statusTag.className = "px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20";
        }
    }
}

async function fetchVisualRAGReferences() {
    try {
        const resp = await fetch("/api/visual_rag/references");
        const refs = await resp.json();
        renderVisualRAGGallery(refs);
    } catch (e) {
        console.error("Failed to fetch visual RAG references:", e);
    }
}

function renderVisualRAGGallery(refs) {
    const listEl = document.getElementById("visual-rag-list");
    const countEl = document.getElementById("visual-rag-count");
    if (countEl) countEl.innerText = `${refs.length} 枚`;
    if (!listEl) return;

    if (refs.length === 0) {
        listEl.innerHTML = `<div class="col-span-full text-center text-slate-500 py-6">登録済みのリファレンス画像はありません。</div>`;
        return;
    }

    listEl.innerHTML = refs.map(r => `
        <div class="bg-slate-950 border ${r.is_anomaly ? 'border-rose-900/60' : 'border-emerald-900/60'} rounded-xl p-2.5 space-y-1.5 relative group">
            <div class="aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center">
                ${safeImageSrc(r.image_base64) ? `<img src="${escapeHtml(safeImageSrc(r.image_base64))}" class="w-full h-full object-cover">` : '<span class="text-[9px] text-slate-600">NO IMAGE</span>'}
            </div>
            <div class="flex items-center justify-between text-[11px]">
                <span class="font-bold text-slate-200 truncate" title="${escapeHtml(r.title)}">${escapeHtml(r.title)}</span>
                <span class="px-1.5 py-0.2 text-[9px] rounded ${r.is_anomaly ? 'bg-rose-500/20 text-rose-300 border border-rose-800' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-800'}">
                    ${r.is_anomaly ? '異常事例' : '正常'}
                </span>
            </div>
            <div class="text-[10px] text-slate-400 truncate">${escapeHtml(r.category)}</div>
            <button data-action="delete-visual-reference" data-id="${escapeHtml(r.ref_id)}" class="absolute top-1.5 right-1.5 bg-black/70 hover:bg-red-600 text-slate-300 hover:text-white rounded-full w-5 h-5 flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition">
                &times;
            </button>
        </div>
    `).join("");
}

async function deleteVisualReference(refId) {
    if (!confirm(`リファレンス画像 '${refId}' を削除しますか？`)) return;
    try {
        await fetch(`/api/visual_rag/references/${encodeURIComponent(refId)}`, { method: "DELETE" });
        fetchVisualRAGReferences();
    } catch (e) {
        console.error("Failed to delete reference:", e);
    }
}

function initVisualRAGModal() {
    const modal = document.getElementById("modal-visual-rag");
    const btnOpen = document.getElementById("btn-open-visual-rag");
    const btnClose = document.getElementById("btn-close-visual-rag");
    const form = document.getElementById("form-register-visual-rag");

    if (btnOpen && modal) btnOpen.addEventListener("click", () => {
        modal.classList.remove("hidden");
        fetchVisualRAGReferences();
    });

    if (btnClose && modal) btnClose.addEventListener("click", () => modal.classList.add("hidden"));

    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            const title = document.getElementById("vr-title").value.trim();
            const category = document.getElementById("vr-category").value;
            const isAnomaly = document.getElementById("vr-anomaly").value === "true";
            const fileInput = document.getElementById("vr-file");

            if (!fileInput.files || fileInput.files.length === 0) {
                alert("画像ファイルを選択してください");
                return;
            }

            const file = fileInput.files[0];
            const reader = new FileReader();
            reader.onload = async () => {
                const b64Data = reader.result;
                try {
                    const resp = await fetch("/api/visual_rag/register", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            title: title,
                            category: category,
                            is_anomaly: isAnomaly,
                            image_base64: b64Data
                        })
                    });
                    if (resp.ok) {
                        form.reset();
                        fetchVisualRAGReferences();
                    } else {
                        const err = await resp.json();
                        alert("登録失敗: " + (err.detail || "エラー"));
                    }
                } catch (err) {
                    alert("通信エラー: " + err);
                }
            };
            reader.readAsDataURL(file);
        });
    }
}
