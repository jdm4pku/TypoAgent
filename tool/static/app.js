const API_BASE = "";
const CHATS_KEY = "typoagent_chats";

const state = {
  sessionId: null,
  apiKey: localStorage.getItem("typoagent_api_key") || "",
  baseUrl: localStorage.getItem("typoagent_base_url") || "",
  lang: localStorage.getItem("typoagent_lang") || "en",
  appearance: localStorage.getItem("typoagent_appearance") || "system",
  accentColor: localStorage.getItem("typoagent_accent") || "default",
  dataStoragePath: localStorage.getItem("typoagent_data_storage") || "default",
  treeReadPath: localStorage.getItem("typoagent_tree_read_path") || "default",
  treePath: localStorage.getItem("typoagent_tree_path") || "",
  treeName: localStorage.getItem("typoagent_tree_name") || "",
  currentChatId: null,
  chats: [],
};

function t(key) {
  return (I18N[state.lang] || I18N.en)[key] ?? key;
}

function applyTranslations() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    const val = t(key);
    if (val) el.textContent = val;
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.getAttribute("data-i18n-placeholder")) || el.placeholder;
  });
  document.querySelectorAll("[data-i18n-html]").forEach((el) => {
    const val = t(el.getAttribute("data-i18n-html"));
    if (val) el.innerHTML = val;
  });
  document.title = t("appTitle");
}

function loadChats() {
  try {
    state.chats = JSON.parse(localStorage.getItem(CHATS_KEY) || "[]");
  } catch (_) {
    state.chats = [];
  }
}

function saveChats() {
  localStorage.setItem(CHATS_KEY, JSON.stringify(state.chats));
}

// Sidebar
const sidebar = document.getElementById("sidebar");
const sidebarRail = document.getElementById("sidebarRail");
document.getElementById("sidebarClose").addEventListener("click", () => {
  sidebar.classList.add("collapsed");
  sidebarRail.classList.add("visible");
});
function openSidebar() {
  sidebar.classList.remove("collapsed");
  sidebarRail.classList.remove("visible");
}
document.getElementById("sidebarOpen")?.addEventListener("click", openSidebar);
document.getElementById("sidebarOpenBtn")?.addEventListener("click", openSidebar);
document.getElementById("railNewChatBtn")?.addEventListener("click", () => {
  createChat();
  openSidebar();
});
document.getElementById("railSearchBtn")?.addEventListener("click", openSidebar);

function deleteChat(chatId, e) {
  if (e) e.stopPropagation();
  state.chats = state.chats.filter((c) => c.id !== chatId);
  saveChats();
  if (state.currentChatId === chatId) {
    state.currentChatId = null;
    state.sessionId = null;
    const chat = state.chats[0];
    if (chat) {
      loadChat(chat.id);
    } else {
      createChat();
    }
  } else {
    renderChatList();
  }
}

function renderChatList() {
  const list = document.getElementById("chatList");
  list.innerHTML = "";
  state.chats.forEach((chat) => {
    const div = document.createElement("div");
    div.className = "chat-list-item" + (chat.id === state.currentChatId ? " active" : "");
    div.dataset.id = chat.id;
    div.innerHTML = `<span class="chat-list-item-title">${escapeHtml(chat.title || "Untitled")}</span><button class="chat-list-item-delete" title="${t("deleteChat")}" aria-label="${t("deleteChat")}">✕</button>`;
    div.addEventListener("click", (e) => {
      if (e.target.closest(".chat-list-item-delete")) return;
      loadChat(chat.id);
    });
    div.querySelector(".chat-list-item-delete").addEventListener("click", (e) => deleteChat(chat.id, e));
    list.appendChild(div);
  });
}

function createChat() {
  const id = "chat_" + Date.now();
  const chat = { id, title: t("newChat"), messages: [], treePath: state.treePath, treeName: state.treeName, createdAt: Date.now() };
  state.chats.unshift(chat);
  saveChats();
  state.currentChatId = id;
  state.sessionId = null;
  renderChatList();
  loadChat(id);
}

function loadChat(id) {
  const chat = state.chats.find((c) => c.id === id);
  if (!chat) return;
  state.currentChatId = id;
  state.sessionId = chat.sessionId || null;
  const messages = document.getElementById("chatMessages");
  messages.innerHTML = "";
  if (chat.messages && chat.messages.length) {
    chat.messages.forEach((m) => appendMessage(m.role, m.content, false, false));
  } else {
    const welcome = document.createElement("div");
    welcome.id = "welcome";
    welcome.className = "welcome";
    welcome.innerHTML = `
      <h1 data-i18n="welcomeTitle">TypoAgent Requirement Elicitation</h1>
      <p data-i18n="welcomeDesc">Describe your initial requirements. TypoAgent will help you supplement and clarify implicit requirements through questions.</p>
    `;
    messages.appendChild(welcome);
    applyTranslations();
  }
  renderChatList();
}

function appendMessage(role, content, isLoading = false, save = true) {
  const chatMessages = document.getElementById("chatMessages");
  const welcome = chatMessages.querySelector("#welcome");
  if (welcome) welcome.style.display = "none";
  const div = document.createElement("div");
  div.className = `message ${role}${isLoading ? " loading" : ""}`;
  const avatar = role === "user" ? "👤" : "◇";
  div.innerHTML = `<div class="message-avatar">${avatar}</div><div class="message-content">${escapeHtml(content || "")}</div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  if (save && state.currentChatId) {
    const chat = state.chats.find((c) => c.id === state.currentChatId);
    if (chat) {
      if (!chat.messages) chat.messages = [];
      chat.messages.push({ role, content });
      if (role === "user" && content && content.length > 50) chat.title = content.slice(0, 50) + "...";
      saveChats();
      renderChatList();
    }
  }
  return div;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML.replace(/\n/g, "<br>");
}

function updateLoadingToContent(el, content) {
  el.classList.remove("loading");
  el.querySelector(".message-content").innerHTML = escapeHtml(content || "").replace(/\n/g, "<br>");
}

document.getElementById("newChatBtn").addEventListener("click", createChat);

// Tree selector
async function fetchTrees() {
  const treeDir = state.treeReadPath || "default";
  const res = await fetch(API_BASE + "/api/trees?tree_dir=" + encodeURIComponent(treeDir));
  const data = await res.json();
  return data.trees || [];
}

const treeSelectorBtn = document.getElementById("treeSelectorBtn");
const treeSelectorDropdown = document.getElementById("treeSelectorDropdown");
const treeList = document.getElementById("treeList");
const selectedTreeName = document.getElementById("selectedTreeName");
const treeSearch = document.getElementById("treeSearch");

treeSelectorBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  treeSelectorDropdown.classList.toggle("open");
  if (treeSelectorDropdown.classList.contains("open")) loadTreeList();
});

document.addEventListener("click", () => treeSelectorDropdown.classList.remove("open"));

async function loadTreeList() {
  const trees = await fetchTrees();
  const filter = (treeSearch.value || "").toLowerCase();
  treeList.innerHTML = "";
  trees
    .filter((t) => !filter || t.name.toLowerCase().includes(filter))
    .forEach((tree) => {
      const div = document.createElement("div");
      div.className = "tree-item" + (tree.path === state.treePath ? " selected" : "");
      div.innerHTML = `<span class="tree-icon">🌳</span><span>${tree.name}</span>`;
      div.addEventListener("click", () => {
        state.treePath = tree.path;
        state.treeName = tree.name;
        localStorage.setItem("typoagent_tree_path", tree.path);
        localStorage.setItem("typoagent_tree_name", tree.name);
        selectedTreeName.textContent = tree.name;
        treeSelectorDropdown.classList.remove("open");
        loadTreeList();
      });
      treeList.appendChild(div);
    });
}

treeSearch.addEventListener("input", loadTreeList);

document.getElementById("constructTreeBtn").addEventListener("click", () => {
  document.getElementById("chat-panel").classList.remove("active");
  document.getElementById("build-panel").classList.add("active");
  treeSelectorDropdown.classList.remove("open");
});

document.getElementById("backToChatBtn").addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  document.getElementById("build-panel").classList.remove("active");
  document.getElementById("chat-panel").classList.add("active");
});

function initTreeSelector() {
  if (state.treeName) {
    selectedTreeName.textContent = state.treeName;
  } else {
    fetchTrees().then((trees) => {
      if (trees.length) {
        state.treePath = trees[0].path;
        state.treeName = trees[0].name;
        selectedTreeName.textContent = trees[0].name;
      } else {
        selectedTreeName.textContent = t("selectTree");
      }
    });
  }
}

// Settings
const settingsModal = document.getElementById("settingsModal");
function syncCustomSelects() {
  document.querySelectorAll(".custom-select").forEach(function(wrap) {
    var sel = document.getElementById(wrap.dataset.select);
    if (!sel) return;
    var trigger = wrap.querySelector(".custom-select-trigger");
    var opt = wrap.querySelector('.custom-select-option[data-value="' + sel.value + '"]');
    wrap.querySelectorAll(".custom-select-option").forEach(function(o) {
      o.classList.toggle("selected", o.dataset.value === sel.value);
    });
    trigger.textContent = opt ? opt.textContent : sel.value;
  });
}
function openSettingsModal() {
  const apiKeyInput = document.getElementById("apiKey");
  const baseUrlInput = document.getElementById("baseUrl");
  apiKeyInput.value = state.apiKey;
  baseUrlInput.value = state.baseUrl;
  document.getElementById("dataStorageSelect").value = state.dataStoragePath;
  document.getElementById("treeReadPathSelect").value = state.treeReadPath;
  document.getElementById("appearanceSelect").value = state.appearance;
  document.getElementById("accentSelect").value = state.accentColor;
  document.getElementById("languageSelect").value = state.lang;
  document.getElementById("testResult").textContent = "";
  document.getElementById("testResult").className = "test-result";
  settingsModal.classList.add("active");
  applyTranslations();
  syncCustomSelects();
}
document.getElementById("settingsBtn").addEventListener("click", openSettingsModal);
document.getElementById("settingsBtnBuild")?.addEventListener("click", openSettingsModal);

function closeSettingsModal() {
  settingsModal.classList.remove("active");
  document.querySelectorAll(".custom-select.open").forEach((w) => w.classList.remove("open"));
}
document.getElementById("settingsClose").addEventListener("click", closeSettingsModal);
settingsModal.addEventListener("click", (e) => {
  if (e.target === settingsModal) closeSettingsModal();
});
const settingsContent = document.getElementById("settingsModal")?.querySelector(".settings-content");
settingsContent?.addEventListener("click", (e) => {
  e.stopPropagation();
  if (!e.target.closest(".custom-select")) {
    document.querySelectorAll(".custom-select.open").forEach((w) => w.classList.remove("open"));
  }
});

document.querySelectorAll(".settings-nav").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".settings-nav").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".settings-section").forEach((s) => s.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("section-" + btn.dataset.section).classList.add("active");
  });
});

document.querySelectorAll(".custom-select").forEach((wrap) => {
  const sel = document.getElementById(wrap.dataset.select);
  if (!sel) return;
  const trigger = wrap.querySelector(".custom-select-trigger");
  const opts = wrap.querySelectorAll(".custom-select-option");
  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    document.querySelectorAll(".custom-select").forEach((w) => w.classList.remove("open"));
    wrap.classList.toggle("open");
  });
  opts.forEach((opt) => {
    opt.addEventListener("click", () => {
      sel.value = opt.dataset.value;
      syncCustomSelects();
      wrap.classList.remove("open");
    });
  });
});
document.addEventListener("click", () => {
  document.querySelectorAll(".custom-select.open").forEach((w) => w.classList.remove("open"));
});
document.querySelectorAll(".custom-select .custom-select-trigger, .custom-select .custom-select-dropdown").forEach((el) => {
  el.addEventListener("click", (e) => e.stopPropagation());
});

function saveAllSettings() {
  state.apiKey = document.getElementById("apiKey").value.trim();
  state.baseUrl = document.getElementById("baseUrl").value.trim();
  state.dataStoragePath = document.getElementById("dataStorageSelect").value;
  state.treeReadPath = document.getElementById("treeReadPathSelect").value;
  state.appearance = document.getElementById("appearanceSelect").value;
  state.accentColor = document.getElementById("accentSelect").value;
  state.lang = document.getElementById("languageSelect").value;
  localStorage.setItem("typoagent_api_key", state.apiKey);
  localStorage.setItem("typoagent_base_url", state.baseUrl);
  localStorage.setItem("typoagent_data_storage", state.dataStoragePath);
  localStorage.setItem("typoagent_tree_read_path", state.treeReadPath);
  localStorage.setItem("typoagent_appearance", state.appearance);
  localStorage.setItem("typoagent_accent", state.accentColor);
  localStorage.setItem("typoagent_lang", state.lang);
  applyTranslations();
  applyAppearance();
  applyAccent();
  closeSettingsModal();
}

document.getElementById("saveGeneralBtn").addEventListener("click", saveAllSettings);
document.getElementById("saveDataBtn").addEventListener("click", saveAllSettings);
document.getElementById("saveSettingsBtn").addEventListener("click", saveAllSettings);

document.getElementById("testApiBtn").addEventListener("click", async () => {
  const apiKey = document.getElementById("apiKey").value.trim();
  const testResult = document.getElementById("testResult");
  if (!apiKey) {
    testResult.textContent = t("fillApiKey");
    testResult.className = "test-result error";
    return;
  }
  const btn = document.getElementById("testApiBtn");
  btn.disabled = true;
  testResult.textContent = t("testing");
  testResult.className = "test-result";
  try {
    const res = await fetch(API_BASE + "/api/test-api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey, base_url: document.getElementById("baseUrl").value.trim() || undefined }),
    });
    const data = await res.json();
    if (data.ok) {
      testResult.textContent = t("testSuccess");
      testResult.className = "test-result success";
    } else {
      testResult.textContent = t("testFailed") + ": " + (data.error || "");
      testResult.className = "test-result error";
    }
  } catch (e) {
    testResult.textContent = t("testFailed") + ": " + e.message;
    testResult.className = "test-result error";
  }
  btn.disabled = false;
});

function applyAppearance() {
  const v = state.appearance;
  const dark = v === "dark" || (v === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.body.classList.toggle("light-theme", !dark);
}
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (state.appearance === "system") applyAppearance();
});

function applyAccent() {
  document.body.dataset.accent = state.accentColor;
}

const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

async function startChat(initialReq) {
  appendMessage("user", initialReq);
  const loadingEl = appendMessage("assistant", "", true, false);
  sendBtn.disabled = true;
  try {
    const res = await fetch(API_BASE + "/api/chat/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        initial_requirements: initialReq,
        api_key: state.apiKey,
        base_url: state.baseUrl || undefined,
        tree_path: state.treePath || undefined,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    state.sessionId = data.session_id;
    const chat = state.chats.find((c) => c.id === state.currentChatId);
    if (chat) {
      chat.sessionId = data.session_id;
      chat.messages.push({ role: "assistant", content: data.response || t("conversationEnded") });
      saveChats();
      renderChatList();
    }
    updateLoadingToContent(loadingEl, data.response || t("conversationEnded"));
  } catch (e) {
    updateLoadingToContent(loadingEl, t("errorPrefix") + e.message);
  }
  sendBtn.disabled = false;
  document.getElementById("chatMessages").scrollTop = 1e9;
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  if (!state.apiKey) {
    settingsModal.classList.add("active");
    document.querySelector('.settings-nav[data-section="api"]').click();
    return;
  }
  appendMessage("user", text);
  chatInput.value = "";
  const loadingEl = appendMessage("assistant", "", true, false);
  sendBtn.disabled = true;
  try {
    const url = state.sessionId ? "/api/chat" : "/api/chat/init";
    const chat = state.chats.find((c) => c.id === state.currentChatId);
    const msgs = (chat && chat.messages) || [];
    const historyBeforeNew = msgs.slice(0, -1);
    const conversation_history = historyBeforeNew.map((m) => ({
      role: m.role === "assistant" ? "interviewer" : m.role,
      content: m.content || "",
    }));
    const body = state.sessionId
      ? { message: text, session_id: state.sessionId, conversation_history, api_key: state.apiKey, base_url: state.baseUrl || undefined, tree_path: state.treePath || undefined }
      : { initial_requirements: text, api_key: state.apiKey, base_url: state.baseUrl || undefined, tree_path: state.treePath || undefined };
    const res = await fetch(API_BASE + url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    if (data.session_id) {
      state.sessionId = data.session_id;
      const chat = state.chats.find((c) => c.id === state.currentChatId);
      if (chat) {
        chat.sessionId = data.session_id;
        chat.messages.push({ role: "assistant", content: data.response || t("conversationEnded") });
        saveChats();
        renderChatList();
      }
    }
    updateLoadingToContent(loadingEl, data.response || t("conversationEnded"));
  } catch (e) {
    updateLoadingToContent(loadingEl, t("errorPrefix") + e.message);
  }
  sendBtn.disabled = false;
  document.getElementById("chatMessages").scrollTop = 1e9;
}

document.querySelector(".chat-add-btn")?.addEventListener("click", () => document.getElementById("chatInput").focus());
sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Build tree (uses API key from state)
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const buildBtn = document.getElementById("buildBtn");
const buildLog = document.getElementById("buildLog");
let uploadedContent = null;

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) readFile(file);
});
fileInput.addEventListener("change", () => { const f = fileInput.files[0]; if (f) readFile(f); });

function readFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadedContent = e.target.result;
    buildBtn.disabled = !uploadedContent || !state.apiKey;
    uploadZone.querySelector(".upload-content p:first-of-type").textContent = t("selectedFile") + file.name;
  };
  reader.readAsText(file, "UTF-8");
}

buildBtn.addEventListener("click", async () => {
  if (!state.apiKey) {
    buildLog.textContent = t("fillApiKey");
    return;
  }
  if (!uploadedContent) {
    buildLog.textContent = t("uploadFirst");
    return;
  }
  buildBtn.disabled = true;
  buildLog.textContent = t("building");
  const formData = new FormData();
  formData.append("content", uploadedContent);
  formData.append("api_key", state.apiKey);
  const treeName = (document.getElementById("treeNameInput").value || "").trim();
  if (treeName) formData.append("tree_name", treeName);
  formData.append("use_layer2", document.getElementById("useLayer2").checked);
  formData.append("use_layer3", document.getElementById("useLayer3").checked);
  formData.append("enable_sampling", document.getElementById("enableSampling").checked);
  try {
    const res = await fetch(API_BASE + "/api/build-tree", { method: "POST", body: formData });
    const data = await res.json();
    buildLog.textContent = res.ok ? t("buildSuccess") + (data.tree_path || "") : t("buildFailed") + (data.error || "");
    if (res.ok) loadTreeList();
  } catch (e) {
    buildLog.textContent = t("requestFailed") + e.message;
  }
  buildBtn.disabled = false;
});

// Init
loadChats();
applyTranslations();
applyAppearance();
applyAccent();
initTreeSelector();
renderChatList();
if (!state.chats.length) createChat();
