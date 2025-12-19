const inputEl = document.getElementById("input-text");
const runBtn = document.getElementById("run-btn");
const statusEl = document.getElementById("status");
const correctedEl = document.getElementById("corrected-text");
const rawEl = document.getElementById("raw-text");
const editListEl = document.getElementById("edit-list");
const sampleBtn = document.getElementById("fill-sample");

const sampleText = "他以为自已很聪明，却经常写错别字。";

const escapeHtml = (str) =>
  str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

const setStatus = (text, busy = false) => {
  statusEl.textContent = text;
  runBtn.disabled = busy;
  runBtn.textContent = busy ? "纠错中..." : "开始纠错";
};

const getSelectedModel = () => {
  const selected = document.querySelector('input[name="model"]:checked');
  return selected ? selected.value : "softmasked";
};

const hasNonReplaceEdits = (edits) => edits.some((edit) => edit.type !== "replace");

const renderHighlighted = (raw, corrected, edits) => {
  if (!edits.length || hasNonReplaceEdits(edits)) {
    return escapeHtml(corrected);
  }
  const marks = new Set(edits.map((edit) => edit.index));
  return corrected
    .split("")
    .map((ch, idx) =>
      marks.has(idx) ? `<span class="hl">${escapeHtml(ch)}</span>` : escapeHtml(ch)
    )
    .join("");
};

const renderEdits = (edits) => {
  if (!edits.length) {
    editListEl.innerHTML = "<li>暂无修改</li>";
    editListEl.classList.add("muted");
    return;
  }
  editListEl.classList.remove("muted");
  const items = edits.map((edit) => {
    if (edit.type === "replace") {
      return `<li>位置 ${edit.index}: ${escapeHtml(edit.src_char)} → ${escapeHtml(edit.pred_char)}</li>`;
    }
    if (edit.type === "insert") {
      return `<li>位置 ${edit.index}: 插入 ${escapeHtml(edit.pred_char)}</li>`;
    }
    return `<li>位置 ${edit.index}: 删除 ${escapeHtml(edit.src_char)}</li>`;
  });
  editListEl.innerHTML = items.join("");
};

const runCorrection = async () => {
  const text = inputEl.value.trim();
  if (!text) {
    setStatus("请先输入文本");
    return;
  }

  setStatus("请求中...", true);
  correctedEl.classList.add("muted");
  rawEl.classList.add("muted");
  correctedEl.textContent = "处理中...";
  rawEl.textContent = "处理中...";
  editListEl.innerHTML = "<li>处理中...</li>";

  try {
    const res = await fetch("/api/correct", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        model: getSelectedModel(),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "请求失败");
    }

    const data = await res.json();
    const highlighted = renderHighlighted(data.raw, data.corrected, data.edits);
    correctedEl.innerHTML = highlighted;
    correctedEl.classList.remove("muted");
    rawEl.textContent = data.raw;
    rawEl.classList.remove("muted");
    renderEdits(data.edits);
    const label =
      data.model === "deepseek"
        ? "DeepSeek"
        : data.model === "qwen"
          ? "Qwen3"
          : "Soft-Masked BERT";
    setStatus(`完成：${label}`);
  } catch (err) {
    setStatus(err.message || "请求失败");
    correctedEl.textContent = "纠错失败";
    rawEl.textContent = text;
    editListEl.innerHTML = "<li>请求失败，请检查后端日志</li>";
  } finally {
    runBtn.disabled = false;
    if (!statusEl.textContent) {
      setStatus("");
    }
  }
};

runBtn.addEventListener("click", runCorrection);
sampleBtn.addEventListener("click", () => {
  inputEl.value = sampleText;
  inputEl.focus();
});
