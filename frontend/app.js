document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predict-form');
  const resultEl = document.getElementById('result');
  const historyEl = document.getElementById('history');
  const urlInput = document.getElementById('url');
  const submitBtn = document.getElementById('submit');

  const state = { last: null };

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.textContent = isLoading ? 'Checking…' : 'Check';
    if (isLoading) submitBtn.classList.add('btn-loading');
    else submitBtn.classList.remove('btn-loading');
  }

  function renderResult(pkg) {
    // pkg: {prediction, score, threshold, proba, features, hyperparams}
    const score = Number(pkg.score ?? 0);
    const threshold = Number(pkg.threshold ?? 0.6);
    const pct = Math.round(score * 100);
    const isMal = score >= threshold;

    const color = isMal ? 'var(--danger)' : 'var(--success)';

    const featureNames = pkg.feature_names || Object.keys(pkg.features || {});

    resultEl.innerHTML = '';

    const card = document.createElement('div');
    card.className = 'result-card';

    const v = document.createElement('div');
    v.className = 'verdict';
    v.textContent = (String(pkg.prediction || (isMal ? 'malicious' : 'benign'))).toUpperCase();
    v.style.color = color;

    const barWrap = document.createElement('div');
    barWrap.className = 'bar';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = pct + '%';
    fill.style.background = color;
    barWrap.appendChild(fill);

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = `Confidence: <strong>${pct}%</strong> (threshold ${Math.round(threshold*100)}%)`;

    const left = document.createElement('div');
    left.appendChild(v);
    left.appendChild(meta);

    card.appendChild(left);
    card.appendChild(barWrap);

    resultEl.appendChild(card);

    // features table
    const featWrap = document.createElement('div');
    featWrap.className = 'features';
    const table = document.createElement('table');
    featureNames.forEach((fn) => {
      const tr = document.createElement('tr');
      const td1 = document.createElement('td');
      td1.textContent = fn;
      const td2 = document.createElement('td');
      td2.textContent = String((pkg.features && pkg.features[fn]) ?? '');
      tr.appendChild(td1);
      tr.appendChild(td2);
      table.appendChild(tr);
    });
    featWrap.appendChild(table);
    resultEl.appendChild(featWrap);

    // actions
    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.gap = '8px';
    actions.style.marginTop = '10px';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'btn ghost';
    copyBtn.textContent = 'Copy result';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(JSON.stringify(pkg, null, 2));
      copyBtn.textContent = 'Copied';
      setTimeout(() => (copyBtn.textContent = 'Copy result'), 1500);
    });

    actions.appendChild(copyBtn);
    resultEl.appendChild(actions);
  }

  function saveHistory(entry) {
    const hist = JSON.parse(localStorage.getItem('phisaviour.history') || '[]');
    hist.unshift(entry);
    localStorage.setItem('phisaviour.history', JSON.stringify(hist.slice(0, 30)));
    renderHistory();
  }

  function renderHistory() {
    const hist = JSON.parse(localStorage.getItem('phisaviour.history') || '[]');
    historyEl.innerHTML = '';
    hist.forEach((h) => {
      const li = document.createElement('li');
      li.className = 'history-item';
      const left = document.createElement('div');
      left.innerHTML = `<div><strong>${h.url}</strong></div><div class="muted">${h.time}</div>`;
      const right = document.createElement('div');
      right.textContent = `${Math.round((h.score||0)*100)}%`;
      right.style.color = h.score >= (h.threshold||0.6) ? 'var(--danger)' : 'var(--success)';
      li.appendChild(left);
      li.appendChild(right);
      historyEl.appendChild(li);
    });
  }

  renderHistory();

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = urlInput.value.trim();
    if (!url) return;

    setLoading(true);
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      let data = null;
      try { data = await res.json(); } catch(_) { data = null; }

      if (!res.ok) {
        // If server returns error, fall back to local predictor
        console.warn('Backend error, falling back to local predictor', data, res.status);
        const local = localPredict(url);
        state.last = local;
        renderResult(local);
        saveHistory({ url, score: local.score, threshold: local.threshold, time: new Date().toLocaleString() });
        setLoading(false);
        return;
      }

      const pkg = {
        prediction: data.prediction ?? data.pred ?? null,
        score: data.score ?? (data.proba ? data.proba[0] : 0),
        threshold: data.threshold ?? 0.6,
        proba: data.proba ?? null,
        features: data.features ?? {},
        feature_names: data.feature_names ?? null,
        hyperparams: data.hyperparams ?? null,
      };

      state.last = pkg;
      renderResult(pkg);
      saveHistory({ url, score: pkg.score, threshold: pkg.threshold, time: new Date().toLocaleString() });
    } catch (err) {
      // Network error -> use local predictor so UI still shows a result
      console.warn('Network error, using local predictor', err);
      const local = localPredict(url);
      state.last = local;
      renderResult(local);
      saveHistory({ url, score: local.score, threshold: local.threshold, time: new Date().toLocaleString() });
    } finally {
      setLoading(false);
    }
  });

  // --- Local predictor fallback (client-side) ---
  function extractFeaturesJS(raw) {
    try {
      const u = new URL(raw);
      const host = u.hostname || '';
      const path = u.pathname || '';
      const url_full = raw;

      const url_length = url_full.length;
      const host_length = host.length;
      const path_length = path.length;
      const num_dots = (host.match(/\./g) || []).length;
      const num_hyphens = (url_full.match(/-/g) || []).length;
      const has_ip = /^\d+\.\d+\.\d+\.\d+$/.test(host) ? 1 : 0;
      const suspicious_words_list = ['login','signin','bank','secure','update','verify','account','confirm','password','wp-login'];
      const url_lc = url_full.toLowerCase();
      let suspicious_words = 0;
      suspicious_words_list.forEach(w => { if (url_lc.includes(w)) suspicious_words += 1; });
      const has_percent_encoded = /%[0-9a-fA-F]{2}/.test(raw) ? 1 : 0;
      const num_subdomains = Math.max(0, host.split('.').length - 2);

      return {
        url_length, host_length, path_length, num_dots, num_hyphens, has_ip,
        suspicious_words, has_percent_encoded, num_subdomains
      };
    } catch (e) {
      // If URL parsing fails, provide safe defaults
      return { url_length: raw.length, host_length: 0, path_length: 0, num_dots:0, num_hyphens:0, has_ip:0, suspicious_words:0, has_percent_encoded:0, num_subdomains:0 };
    }
  }

  function sigmoid(x){ return 1/(1+Math.exp(-x)); }

  function localPredict(url){
    const feats = extractFeaturesJS(url);
    // mirror server heuristic: higher is more suspicious
    let raw_score = (feats.suspicious_words * 2.0) + (feats.num_hyphens * 0.5) + (feats.num_subdomains * 0.3);
    raw_score += (50 - Math.min(50, feats.url_length)) * -0.01;
    const prob = sigmoid(raw_score);
    const threshold = 0.6;
    const prediction = prob >= threshold ? 'malicious' : 'benign';
    return { prediction, score: prob, threshold, proba: [prob, 1-prob], features: feats };
  }
});
