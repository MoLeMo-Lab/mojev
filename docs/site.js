const tabs = [...document.querySelectorAll('[role="tab"][data-tab]')];
const panels = [...document.querySelectorAll('[role="tabpanel"][data-panel]')];

document.querySelectorAll('[data-math-display]').forEach((element) => {
  katex.render(element.textContent.trim(), element, {
    displayMode: true,
    throwOnError: true,
    trust: false,
  });
});

function activateTab(tab, focus = false) {
  tabs.forEach((item) => {
    const active = item === tab;
    item.setAttribute('aria-selected', String(active));
    item.tabIndex = active ? 0 : -1;
  });
  panels.forEach((panel) => { panel.hidden = panel.dataset.panel !== tab.dataset.tab; });
  if (focus) tab.focus();
}

tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', (event) => {
    let next = index;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault();
    activateTab(tabs[next], true);
  });
});

const copyButton = document.getElementById('copy-citation');
copyButton?.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex')?.textContent ?? '';
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = 'Copied';
    if (status) status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    copyButton.textContent = 'Select BibTeX below';
    if (status) status.textContent = 'Copy unavailable; select the citation text below.';
  }
});
