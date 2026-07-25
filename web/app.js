const els = {
  prompt: document.querySelector("#prompt"),
  systemPrompt: document.querySelector("#systemPrompt"),
  modelA: document.querySelector("#modelA"),
  modelB: document.querySelector("#modelB"),
  temperature: document.querySelector("#temperature"),
  temperatureValue: document.querySelector("#temperatureValue"),
  topP: document.querySelector("#topP"),
  topPValue: document.querySelector("#topPValue"),
  maxTokens: document.querySelector("#maxTokens"),
  runButton: document.querySelector("#runButton"),
  status: document.querySelector("#status"),
  cheaper: document.querySelector("#cheaper"),
  faster: document.querySelector("#faster"),
  totalCost: document.querySelector("#totalCost"),
  totalTokens: document.querySelector("#totalTokens"),
};

const slots = [
  {
    title: document.querySelector("#titleA"),
    badge: document.querySelector("#badgeA"),
    latency: document.querySelector("#latencyA"),
    tokens: document.querySelector("#tokensA"),
    cost: document.querySelector("#costA"),
    answer: document.querySelector("#answerA"),
  },
  {
    title: document.querySelector("#titleB"),
    badge: document.querySelector("#badgeB"),
    latency: document.querySelector("#latencyB"),
    tokens: document.querySelector("#tokensB"),
    cost: document.querySelector("#costB"),
    answer: document.querySelector("#answerB"),
  },
];

function money(value) {
  return `$${Number(value || 0).toFixed(6)}`;
}

function seconds(value) {
  return `${Number(value || 0).toFixed(2)}s`;
}

function setBusy(isBusy) {
  els.runButton.disabled = isBusy;
  els.runButton.innerHTML = isBusy
    ? '<span class="button-icon">...</span>Running'
    : '<span class="button-icon">&gt;</span>Compare';
}

function syncRanges() {
  els.temperatureValue.value = Number(els.temperature.value).toFixed(1);
  els.topPValue.value = Number(els.topP.value).toFixed(2);
}

async function loadConfig() {
  const res = await fetch("/api/config");
  const config = await res.json();
  els.modelA.value = config.models[0] || "gpt-4o";
  els.modelB.value = config.models[1] || "gpt-4o-mini";
}

function render(data) {
  let totalCost = 0;
  let totalTokens = 0;

  data.results.forEach((result, index) => {
    const slot = slots[index];
    const promptTokens = result.cost.prompt_tokens || 0;
    const completionTokens = result.cost.completion_tokens || 0;
    const tokens = promptTokens + completionTokens;
    totalCost += result.cost.total_cost || 0;
    totalTokens += tokens;

    slot.title.textContent = result.model;
    slot.latency.textContent = seconds(result.latency);
    slot.tokens.textContent = tokens.toLocaleString();
    slot.cost.textContent = money(result.cost.total_cost);
    slot.answer.textContent = result.answer || "(empty response)";
    slot.badge.textContent =
      result.model === data.winner.cheaper && result.model === data.winner.faster
        ? "Cheaper + faster"
        : result.model === data.winner.cheaper
          ? "Cheaper"
          : result.model === data.winner.faster
            ? "Faster"
            : "Compared";
  });

  els.cheaper.textContent = data.winner.cheaper;
  els.faster.textContent = data.winner.faster;
  els.totalCost.textContent = money(totalCost);
  els.totalTokens.textContent = totalTokens.toLocaleString();
}

async function compare() {
  els.status.textContent = "";
  setBusy(true);
  slots.forEach((slot) => {
    slot.badge.textContent = "Running";
    slot.answer.textContent = "";
  });

  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: els.prompt.value,
        system_prompt: els.systemPrompt.value,
        models: [els.modelA.value.trim(), els.modelB.value.trim()],
        temperature: Number(els.temperature.value),
        top_p: Number(els.topP.value),
        max_tokens: Number(els.maxTokens.value),
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Comparison failed.");
    }
    render(data);
  } catch (err) {
    els.status.textContent = err.message;
    slots.forEach((slot) => {
      slot.badge.textContent = "Error";
    });
  } finally {
    setBusy(false);
  }
}

els.temperature.addEventListener("input", syncRanges);
els.topP.addEventListener("input", syncRanges);
els.runButton.addEventListener("click", compare);

syncRanges();
loadConfig().catch((err) => {
  els.status.textContent = err.message;
});
