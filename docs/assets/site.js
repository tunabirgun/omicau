(() => {
  const root = document.documentElement;
  root.classList.add('js');
  document.querySelector('.appearance').hidden = false;
  const modes = [...document.querySelectorAll('[data-theme-choice]')];
  let saved;
  try { saved = localStorage.getItem('omicau-theme'); } catch {}
  const setTheme = theme => {
    root.dataset.theme = theme;
    modes.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.themeChoice === theme)));
  };
  setTheme(saved === 'dark' ? 'dark' : 'light');
  modes.forEach(button => button.addEventListener('click', () => {
    setTheme(button.dataset.themeChoice);
    try { localStorage.setItem('omicau-theme', root.dataset.theme); } catch {}
  }));

  const menu = document.querySelector('#menu-button');
  menu.hidden = false;
  const navigation = document.querySelector('#site-navigation');
  const closeMenu = () => { menu.setAttribute('aria-expanded', 'false'); navigation.classList.remove('is-open'); };
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') !== 'true';
    menu.setAttribute('aria-expanded', String(open));
    navigation.classList.toggle('is-open', open);
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') { closeMenu(); menu.focus(); }
  });
  navigation.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));

  document.querySelectorAll('[data-copy]').forEach(button => {
    button.hidden = !navigator.clipboard;
    button.addEventListener('click', async () => {
      const code = document.getElementById(button.dataset.copy);
      const status = button.parentElement.querySelector('[role="status"]');
      try { await navigator.clipboard.writeText(code.textContent); status.textContent = 'Copied.'; }
      catch { status.textContent = 'Select the commands to copy them.'; }
    });
  });

  if (document.body.dataset.page === 'home') {
    const routes = { '#evidence': 'guide.html#workflow', '#safeguards': 'interpretation.html#checks', '#quickstart': 'guide.html#quickstart', '#inputs': 'guide.html#inputs', '#reproducibility': 'reproducibility.html', '#repro': 'reproducibility.html', '#limitations': 'interpretation.html#scope', '#links': 'reproducibility.html#resources' };
    if (routes[location.hash]) { location.replace(routes[location.hash]); return; }
  }

  const select = document.querySelector('#example-select');
  if (!select) return;
  select.disabled = false;
  // Constructed examples for the website; these are not study measurements.
  const examples = {
    increase: { a: .78, b: .72, full: .84, interval: [.02, .10], text: 'Including assay B improves this configured fusion model. Read its contribution alongside the standalone scores.' },
    unresolved: { a: .81, b: .74, full: .82, interval: [-.02, .04], text: 'The interval crosses zero. This comparison leaves the contribution of assay B unresolved.' },
    reduction: { a: .86, b: .76, full: .82, interval: [-.07, -.01], text: 'Assay B predicts the endpoint on its own, but including it reduces performance in this configured fusion model.' }
  };
  const signed = value => (value > 0 ? '+' : '') + value.toFixed(2);
  function render() {
    const item = examples[select.value];
    const delta = item.full - item.a;
    const [lower, upper] = item.interval;
    const category = lower > 0 ? 'Observed increase' : upper < 0 ? 'Observed reduction' : 'Unresolved';
    document.querySelector('#example-chart').innerHTML = `
      <title>Constructed comparison of standalone and fusion scores</title>
      <desc>Assay A alone ${item.a.toFixed(2)}; assay B alone ${item.b.toFixed(2)}; full fusion ${item.full.toFixed(2)}. Scores are illustrative AUROC values.</desc>
      ${[.5,.6,.7,.8,.9,1].map(v => `<line class="gridline" x1="${170+(v-.5)*650}" x2="${170+(v-.5)*650}" y1="25" y2="204"/><text class="tick" x="${170+(v-.5)*650}" y="232" text-anchor="middle">${v.toFixed(1)}</text>`).join('')}
      ${[['Assay A alone',item.a],['Assay B alone',item.b],['Full fusion',item.full]].map(([label,value],i) => `<text class="plot-label" x="147" y="${61+i*62}" text-anchor="end">${label}</text><line class="score-line" x1="170" x2="${170+(value-.5)*650}" y1="${56+i*62}" y2="${56+i*62}"/><circle class="score-point ${i===2?'fusion-point':''}" cx="${170+(value-.5)*650}" cy="${56+i*62}" r="5.5"/><text class="score-value" x="540" y="${61+i*62}" text-anchor="end">${value.toFixed(2)}</text>`).join('')}
      <text class="axis-title" x="330" y="267" text-anchor="middle">Held-out score (AUROC)</text>`;
    document.querySelector('#difference-value').textContent = signed(delta);
    document.querySelector('#interval-value').textContent = `${signed(lower)} to ${signed(upper)}`;
    document.querySelector('#category-value').textContent = category;
    document.querySelector('#example-explanation').textContent = item.text;
    document.querySelector('#example-result').textContent = `${category}. Full fusion minus assay A alone: ${signed(delta)}. Illustrative interval: ${signed(lower)} to ${signed(upper)}.`;
    document.querySelectorAll('[data-score]').forEach(cell => { cell.textContent = item[cell.dataset.score].toFixed(2); });
  }
  select.addEventListener('change', render);
  render();
})();
