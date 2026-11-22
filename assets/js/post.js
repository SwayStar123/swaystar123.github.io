/*
 * Post pages: dresses the HTML kramdown produces so the Markdown itself can
 * stay plain. Runs before cave.js, which then measures the finished page.
 *
 *   - section headings   numbered rules, whatever heading level the post uses
 *   - image paragraphs   <figure> with a numbered caption taken from the alt text
 *   - tables             a scroll box, and right-aligned numeric columns
 *   - contents rail      built from the section headings, tracks the scroll
 */
(function () {
  'use strict';

  var body = document.getElementById('post-body');
  if (!body) return;
  var all = function (sel, root) { return [].slice.call((root || body).querySelectorAll(sel)); };

  /* ---- headings ----
     Posts don't agree on which level a section is (#, ## or ###), so style by
     depth relative to the shallowest heading the post actually uses. */

  var heads = all('h1, h2, h3, h4, h5, h6').filter(function (h) { return !h.closest('.footnotes'); });
  var shallowest = heads.reduce(function (min, h) { return Math.min(min, +h.tagName.charAt(1)); }, 7);
  var sections = [];
  heads.forEach(function (h) {
    var depth = +h.tagName.charAt(1) - shallowest;
    h.classList.add(depth === 0 ? 'h-sec' : depth === 1 ? 'h-sub' : 'h-min');
    if (depth !== 0) return;
    // .h-sec is a flex row (number, title, rule): keep the title one item even
    // when it contains code or links
    var title = document.createElement('span');
    while (h.firstChild) title.appendChild(h.firstChild);
    h.appendChild(title);
    sections.push(h);
  });

  /* ---- figures ---- */

  var figures = 0;
  all('p').forEach(function (p) {
    // a paragraph that is nothing but one image, linked or not
    if (p.children.length !== 1 || p.textContent.trim() !== '') return;
    var only = p.firstElementChild;
    var img = only.tagName === 'IMG' ? only
      : only.tagName === 'A' && only.children.length === 1 && only.firstElementChild.tagName === 'IMG' ? only.firstElementChild
      : null;
    if (!img) return;

    var fig = document.createElement('figure');
    fig.className = 'fig';
    var cap = document.createElement('figcaption');
    var no = document.createElement('span');
    no.className = 'fig__no';
    no.textContent = 'FIG ' + (++figures);
    cap.appendChild(no);
    if (img.alt) cap.appendChild(document.createTextNode(img.alt));

    // "(rows: original / noised / reconstruction)" straight after an image is
    // its legend: fold it into the caption, without the brackets
    var next = p.nextElementSibling;
    if (next && next.tagName === 'P' && /^\([\s\S]*\)$/.test(next.textContent.trim())) {
      var legend = document.createElement('span');
      legend.className = 'fig__legend';
      while (next.firstChild) legend.appendChild(next.firstChild);
      var first = legend.firstChild, last = legend.lastChild;
      if (first && first.nodeType === 3 && last && last.nodeType === 3) {
        first.nodeValue = first.nodeValue.replace(/^\s*\(/, '');
        last.nodeValue = last.nodeValue.replace(/\)\s*$/, '');
      }
      cap.appendChild(legend);
      next.parentNode.removeChild(next);
    }

    p.parentNode.replaceChild(fig, p);
    fig.appendChild(only);
    fig.appendChild(cap);
  });

  /* ---- tables ---- */

  var NUMERIC = /^[-+−–]?[\d.,]+(\s*[±x×]\s*[\d.,]+)?\s*(%|[a-zA-Zµ]{1,3})?$/;
  all('table').forEach(function (table) {
    if (table.closest('.highlight')) return;   // rouge's line-number layout, not data
    var wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    table.parentNode.insertBefore(wrap, table);
    wrap.appendChild(table);

    var rows = table.tBodies.length ? [].slice.call(table.tBodies[0].rows) : [];
    if (!rows.length) return;
    for (var c = 0; c < rows[0].cells.length; c++) {
      var numeric = rows.every(function (r) { return r.cells[c] && NUMERIC.test(r.cells[c].textContent.trim()); });
      if (!numeric) continue;
      [].slice.call(table.rows).forEach(function (r) { if (r.cells[c]) r.cells[c].classList.add('num'); });
    }
  });

  /* ---- contents rail ---- */

  var rail = document.getElementById('rail');
  var list = document.getElementById('rail-list');
  if (!rail || !list || sections.length < 3) return;

  var links = sections.map(function (h, k) {
    if (!h.id) h.id = 'section-' + (k + 1);
    var li = document.createElement('li');
    var a = document.createElement('a');
    var label = document.createElement('span');
    a.href = '#' + h.id;
    label.textContent = h.textContent;
    a.appendChild(label);
    li.appendChild(a);
    list.appendChild(li);
    return a;
  });
  rail.hidden = false;

  // the current section is the last one whose heading has passed the upper
  // third of the window
  var queued = false;
  function mark() {
    queued = false;
    var line = innerHeight * 0.3, current = 0;
    for (var k = 0; k < sections.length; k++) {
      if (sections[k].getBoundingClientRect().top <= line) current = k;
    }
    links.forEach(function (a, k) { a.classList.toggle('is-current', k === current); });
  }
  function queue() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(mark);
  }
  addEventListener('scroll', queue, { passive: true });
  addEventListener('resize', queue);
  mark();
})();
