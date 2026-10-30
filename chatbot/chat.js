/* Resume chatbot widget.
 *
 * Include with:  <script src="chatbot/chat.js" data-theme="dashboard|rpg|avatar" [data-mount="elementId"] defer></script>
 * Without data-mount it shows a floating chat bubble; with it, the chat is drawn inside that element.
 * The model itself (chatbot/model.js) is produced by chatbot/train.py and loaded automatically.
 * The widget announces its state with a `resumechat` window event so the avatar can react.
 */
(() => {
  const script = document.currentScript;
  const base = script.src.replace(/[^/]*$/, '');
  const theme = script.dataset.theme || 'dashboard';
  const mountId = script.dataset.mount;
  const EMAIL = 'harshini.g.official@gmail.com';

  /* ---------------- model: must match chatbot/train.py exactly ---------------- */
  let model = null;

  function normalise(text){
    text = text.toLowerCase().replace(/’/g, "'").replace(/'/g, '');
    text = text.replace(/[^a-z0-9+#]+/g, ' ');
    const words = [];
    for (const w of text.split(' ')) {
      if (!w) continue;
      const mapped = model.normalise[w] || w;
      words.push(...mapped.split(' '));
    }
    return words;
  }
  function fnv1a(s){
    let h = 0x811c9dc5;
    for (const byte of new TextEncoder().encode(s)) {
      h ^= byte;
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h >>> 0;
  }
  function featurise(text){
    const D = model.dim, x = new Map();
    const add = (key, v) => { const i = fnv1a(key) % D; x.set(i, (x.get(i) || 0) + v); };
    const words = normalise(text);
    words.forEach((w, i) => {
      add('w|' + w, 1);
      if (i + 1 < words.length) add('b|' + w + ' ' + words[i + 1], 1);
      const padded = '<' + w + '>';
      for (let j = 0; j + 3 <= padded.length; j++) add('c|' + padded.slice(j, j + 3), model.charWeight);
    });
    let norm = 0;
    for (const [i, v] of x) { const lv = Math.log1p(v); x.set(i, lv); norm += lv * lv; }
    norm = Math.sqrt(norm) || 1;
    for (const [i, v] of x) x.set(i, v / norm);
    return x;
  }
  function classify(text){
    const x = featurise(text), C = model.labels.length, D = model.dim, W = model.W;
    const logits = new Float64Array(C);
    for (let c = 0; c < C; c++) {
      let s = model.bias[c];
      const row = c * D, sc = model.scales[c];
      for (const [i, v] of x) s += W[row + i] * sc * v;
      logits[c] = s;
    }
    const m = Math.max(...logits);
    let sum = 0;
    const p = Array.from(logits, l => { const e = Math.exp(l - m); sum += e; return e; }).map(e => e / sum);
    const k = p.indexOf(Math.max(...p));
    const label = model.labels[k];
    const known = label !== 'out_of_scope' && p[k] >= model.threshold && normalise(text).length > 0;
    return {label: known ? label : null, confidence: p[k]};
  }

  /* ---------------- styles ---------------- */
  const themes = {
    dashboard: {bg: '#161B22', panel: '#0D1117', line: '#30363D', text: '#E6EDF3', muted: '#8B949E', accent: '#58A6FF',
                accentText: '#0D1117', bot: '#1C2330', user: '#1F6FEB', userText: '#fff', font: "'Inter', system-ui, sans-serif",
                head: "'JetBrains Mono', monospace", radius: '12px'},
    rpg:       {bg: '#161331', panel: '#0B0A18', line: '#3A2E7A', text: '#F2EFE3', muted: '#9691C4', accent: '#38E1C6',
                accentText: '#0B0A18', bot: '#1D1A42', user: '#FF4D8D', userText: '#0B0A18', font: "'Space Mono', monospace",
                head: "'Press Start 2P', monospace", radius: '2px'},
    avatar:    {bg: '#FFFFFF', panel: '#FFF8F7', line: '#F6C9CF', text: '#3A2422', muted: '#8C6E6B', accent: '#E8768A',
                accentText: '#fff', bot: '#FBEDEE', user: '#E8768A', userText: '#fff', font: "'Inter', system-ui, sans-serif",
                head: "'Inter', system-ui, sans-serif", radius: '16px'},
  };
  const t = themes[theme] || themes.dashboard;
  const css = `
  .rc{--rc-bg:${t.bg}; --rc-panel:${t.panel}; --rc-line:${t.line}; --rc-text:${t.text}; --rc-muted:${t.muted};
      --rc-accent:${t.accent}; --rc-accent-text:${t.accentText}; --rc-bot:${t.bot}; --rc-user:${t.user}; --rc-user-text:${t.userText};
      --rc-radius:${t.radius}; font-family:${t.font}; color:var(--rc-text); font-size:14px; line-height:1.45;}
  .rc *{box-sizing:border-box;}
  .rc-fab{position:fixed; right:20px; bottom:20px; z-index:200; display:flex; align-items:center; gap:8px;
      background:var(--rc-accent); color:var(--rc-accent-text); border:0; border-radius:999px; padding:12px 16px;
      font:600 13px ${t.font}; cursor:pointer; box-shadow:0 10px 26px rgba(0,0,0,.35);}
  .rc[data-theme="rpg"] .rc-fab{border-radius:0; font:9px ${t.head}; box-shadow:0 0 0 3px var(--rc-line), 6px 6px 0 rgba(0,0,0,.5);}
  .rc-fab:hover{filter:brightness(1.08);}
  .rc-panel{display:flex; flex-direction:column; background:var(--rc-bg); border:1px solid var(--rc-line);
      border-radius:var(--rc-radius); overflow:hidden;}
  .rc.floating .rc-panel{position:fixed; right:20px; bottom:20px; z-index:201; width:min(380px, calc(100vw - 32px));
      height:min(560px, calc(100vh - 40px)); box-shadow:0 18px 50px rgba(0,0,0,.45); display:none;}
  .rc.floating.open .rc-panel{display:flex;}
  .rc.floating.open .rc-fab{display:none;}
  .rc.inline .rc-panel{height:100%; min-height:420px;}
  .rc[data-theme="rpg"] .rc-panel{border:3px solid var(--rc-line);}
  .rc-head{display:flex; align-items:center; justify-content:space-between; gap:10px; padding:12px 14px;
      background:var(--rc-panel); border-bottom:1px solid var(--rc-line);}
  .rc-title{font:700 13px ${t.head}; margin:0;}
  .rc[data-theme="rpg"] .rc-title{font-size:9px; letter-spacing:.03em;}
  .rc-sub{font-size:11.5px; color:var(--rc-muted); margin-top:2px;}
  .rc-close{background:none; border:0; color:var(--rc-muted); font-size:20px; line-height:1; cursor:pointer; padding:4px;}
  .rc-close:hover{color:var(--rc-text);}
  .rc.inline .rc-close{display:none;}
  .rc-log{flex:1; overflow-y:auto; padding:14px; display:flex; flex-direction:column; gap:10px;}
  .rc-msg{max-width:86%; padding:9px 12px; border-radius:var(--rc-radius); white-space:pre-wrap; word-wrap:break-word;}
  .rc-msg.bot{background:var(--rc-bot); align-self:flex-start; border-bottom-left-radius:4px;}
  .rc-msg.user{background:var(--rc-user); color:var(--rc-user-text); align-self:flex-end; border-bottom-right-radius:4px;}
  .rc[data-theme="rpg"] .rc-msg{border-radius:0;}
  .rc-msg a{color:inherit; font-weight:600;}
  .rc-msg.bot a{color:var(--rc-accent);}
  .rc-typing{display:inline-flex; gap:4px; padding:12px 14px;}
  .rc-typing i{width:6px; height:6px; border-radius:50%; background:var(--rc-muted); animation:rc-dot 1s infinite;}
  .rc-typing i:nth-child(2){animation-delay:.15s;} .rc-typing i:nth-child(3){animation-delay:.3s;}
  @keyframes rc-dot{0%,60%,100%{opacity:.3; transform:translateY(0);} 30%{opacity:1; transform:translateY(-3px);}}
  .rc-chips{display:flex; flex-wrap:wrap; gap:6px; align-self:flex-start;}
  .rc-chip{background:transparent; color:var(--rc-accent); border:1px solid var(--rc-accent); border-radius:999px;
      padding:5px 10px; font:500 12px ${t.font}; cursor:pointer; text-align:left;}
  .rc[data-theme="rpg"] .rc-chip{border-radius:0;}
  .rc-chip:hover{background:var(--rc-accent); color:var(--rc-accent-text);}
  .rc-form{display:flex; gap:8px; padding:10px; border-top:1px solid var(--rc-line); background:var(--rc-panel);}
  .rc-input{flex:1; min-width:0; background:var(--rc-bg); color:var(--rc-text); border:1px solid var(--rc-line);
      border-radius:var(--rc-radius); padding:10px 12px; font:14px ${t.font}; outline:none;}
  .rc-input:focus{border-color:var(--rc-accent);}
  .rc-send{background:var(--rc-accent); color:var(--rc-accent-text); border:0; border-radius:var(--rc-radius);
      padding:0 14px; font:600 13px ${t.font}; cursor:pointer;}
  .rc-send:disabled{opacity:.5; cursor:default;}
  .rc-foot{font-size:11px; color:var(--rc-muted); padding:0 12px 10px; background:var(--rc-panel);}
  @media (max-width:480px){
    .rc.floating .rc-panel{right:0; bottom:0; width:100vw; height:min(85vh, 600px); border-radius:var(--rc-radius) var(--rc-radius) 0 0;}
  }
  @media (prefers-reduced-motion:reduce){ .rc-typing i{animation:none;} }`;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  /* ---------------- markup ---------------- */
  const root = document.createElement('div');
  root.className = 'rc ' + (mountId ? 'inline' : 'floating');
  root.dataset.theme = theme;
  root.innerHTML = `
    ${mountId ? '' : `<button class="rc-fab" type="button" aria-label="Chat with Harshini's AI twin">💬 Ask me anything</button>`}
    <section class="rc-panel" aria-label="Chat about Harshini's resume">
      <div class="rc-head">
        <div><p class="rc-title">Ask me anything</p><div class="rc-sub">Harshini's AI twin · trained on my resume</div></div>
        <button class="rc-close" type="button" aria-label="Close chat">×</button>
      </div>
      <div class="rc-log" role="log" aria-live="polite"></div>
      <form class="rc-form" autocomplete="off">
        <input class="rc-input" type="text" maxlength="300" placeholder="Type a question…" aria-label="Your question">
        <button class="rc-send" type="submit">Send</button>
      </form>
      <div class="rc-foot">Answers come from a small model trained on Harshini's resume.</div>
    </section>`;
  (mountId ? document.getElementById(mountId) : document.body).appendChild(root);
  const log = root.querySelector('.rc-log'), form = root.querySelector('.rc-form');
  const input = root.querySelector('.rc-input'), send = root.querySelector('.rc-send');

  const emit = (state, intent) => window.dispatchEvent(new CustomEvent('resumechat', {detail: {state, intent}}));
  const esc = s => s.replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  function linkify(text){
    return esc(text)
      .replace(/[\w.+-]+@[\w-]+\.[\w.]+/g, m => `<a href="mailto:${m}">${m}</a>`)
      .replace(/linkedin\.com\/in\/[\w-]+/g, m => `<a href="https://${m}" target="_blank" rel="noopener">${m}</a>`)
      .replace(/\+61[\d ]{9,12}\d/g, m => `<a href="tel:${m.replace(/ /g, '')}">${m}</a>`);
  }
  function scroll(){ log.scrollTop = log.scrollHeight; }
  function bubble(html, who){
    const el = document.createElement('div');
    el.className = 'rc-msg ' + who;
    el.innerHTML = html;
    log.appendChild(el);
    scroll();
    return el;
  }
  function chips(list){
    log.querySelectorAll('.rc-chips').forEach(c => c.remove());
    if (!list || !list.length) return;
    const wrap = document.createElement('div');
    wrap.className = 'rc-chips';
    list.forEach(q => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'rc-chip'; b.textContent = q;
      b.addEventListener('click', () => ask(q));
      wrap.appendChild(b);
    });
    log.appendChild(wrap);
    scroll();
  }

  let busy = false;
  function ask(question){
    question = question.trim();
    if (!question || busy || !model) return;
    busy = true; send.disabled = true;
    chips(null);
    bubble(esc(question), 'user');
    input.value = '';
    const typing = bubble('<span class="rc-typing" aria-label="typing"><i></i><i></i><i></i></span>', 'bot');
    typing.style.padding = '0';
    emit('thinking');
    const {label} = classify(question);
    setTimeout(() => {
      typing.remove();
      if (label) {
        const a = model.answers[label];
        bubble(linkify(a.answer), 'bot');
        chips(a.follow);
        emit('answered', label);
      } else {
        const subject = encodeURIComponent('Question from your resume site');
        const body = encodeURIComponent(`Hi Harshini,\n\nI had a question your resume assistant couldn't answer:\n"${question}"\n\n`);
        bubble(`Sorry! I'm still in training 🙈 For this one, please reach out to me at ` +
               `<a href="mailto:${EMAIL}?subject=${subject}&body=${body}">${EMAIL}</a> and I'll get back to you.`, 'bot');
        chips(['What can I ask you?']);
        emit('unsure');
      }
      busy = false; send.disabled = false;
      if (document.activeElement !== input && matchMedia('(pointer:fine)').matches) input.focus();
    }, 550 + Math.min(question.length * 12, 600));
  }
  form.addEventListener('submit', e => { e.preventDefault(); ask(input.value); });

  function greet(){
    if (log.childElementCount) return;
    bubble("Hi! 👋 I'm Harshini — well, a little AI version of me. Ask me about my experience, skills, education or availability, or tap a question below.", 'bot');
    chips(model.starters);
  }

  /* ---------------- open / close (floating mode) ---------------- */
  if (!mountId) {
    const fab = root.querySelector('.rc-fab');
    const open = () => { root.classList.add('open'); greet(); input.focus(); };
    const close = () => { root.classList.remove('open'); fab.focus(); };
    fab.addEventListener('click', open);
    root.querySelector('.rc-close').addEventListener('click', close);
    root.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
  }

  /* ---------------- load the trained model ---------------- */
  function ready(){
    const m = window.RESUME_CHAT_MODEL;
    const bytes = Uint8Array.from(atob(m.weights), c => c.charCodeAt(0));
    model = Object.assign({}, m, {W: new Int8Array(bytes.buffer)});
    window.ResumeChat = {ask, classify};
    if (mountId) greet();
  }
  if (window.RESUME_CHAT_MODEL) ready();
  else {
    const s = document.createElement('script');
    s.src = base + 'model.js';
    s.onload = ready;
    s.onerror = () => { send.disabled = true; input.placeholder = 'Chat is unavailable right now'; };
    document.head.appendChild(s);
  }
})();
