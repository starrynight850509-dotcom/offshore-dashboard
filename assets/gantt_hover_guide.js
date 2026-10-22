/* F2 Gantt lane guide: drawn in Plotly SVG below the hover tooltip. */
(function () {
  'use strict';
  const ROOT_ID = 'gantt-chart';
  const FLAG = '__ganttLaneGuideV4';
  const NS = 'http://www.w3.org/2000/svg';

  function install() {
    const root = document.getElementById(ROOT_ID);
    const gd = root && root.querySelector('.js-plotly-plot');
    if (!gd || !gd._fullLayout || typeof gd.on !== 'function') return;
    if (gd[FLAG]) return;
    gd[FLAG] = true;

    // Older HTML overlays can cover Plotly hover labels; remove them.
    gd.querySelectorAll('.gantt-lane-hover-guide').forEach(el => el.remove());

    let line = null;
    function ensureLine() {
      const hover = gd.querySelector('svg.main-svg .hoverlayer');
      if (!hover || !hover.parentNode) return null;
      if (line && line.parentNode === hover.parentNode && line.nextSibling === hover) return line;
      if (line) line.remove();
      line = document.createElementNS(NS, 'line');
      line.classList.add('gantt-lane-hover-guide-svg');
      line.setAttribute('stroke', 'rgba(100,116,139,0.25)');
      line.setAttribute('stroke-width', '1');
      line.setAttribute('pointer-events', 'none');
      line.style.display = 'none';
      // This makes the line appear above chart data but BELOW the tooltip.
      hover.parentNode.insertBefore(line, hover);
      return line;
    }

    const hide = () => { if (line) line.style.display = 'none'; };
    function bounds() {
      const l = gd._fullLayout;
      const x = l && l.xaxis;
      const y = l && l.yaxis;
      if (!x || !y) return null;
      return {left:x._offset, right:x._offset+x._length,
        top:y._offset, bottom:y._offset+y._length, y};
    }
    function draw(clientY) {
      const b = bounds();
      const el = ensureLine();
      if (!b || !el) return hide();
      const y = clientY - gd.getBoundingClientRect().top;
      if (y < b.top - 3 || y > b.bottom + 3) return hide();
      el.setAttribute('x1', b.left);
      el.setAttribute('x2', b.right);
      el.setAttribute('y1', y);
      el.setAttribute('y2', y);
      el.style.display = 'block';
    }
    function annotationCenter(target) {
      const a = target.closest('.annotation');
      if (!a) return null;
      const data = gd._fullLayout && gd._fullLayout.annotations;
      if (!data) return null;
      const annotations = Array.from(gd.querySelectorAll('.annotation'));
      const index = annotations.indexOf(a);
      if (index < 0 || !data[index] || data[index].yref !== 'y') return null;
      const r = a.getBoundingClientRect();
      return r.top + r.height / 2;
    }
    gd.addEventListener('pointermove', function (e) {
      const t = e.target;
      if (!(t instanceof Element)) return hide();
      const a = annotationCenter(t);
      if (a !== null) return draw(a);
      if (t.closest('.nsewdrag, .draglayer, .cartesianlayer, .plot')) return draw(e.clientY);
      hide();
    });
    gd.addEventListener('pointerleave', hide);
    gd.on('plotly_hover', function (evt) {
      const p = evt && evt.points && evt.points[0];
      const b = bounds();
      if (!p || !b || p.yaxis !== b.y || p.y == null) return;
      draw(gd.getBoundingClientRect().top + b.y._offset + b.y.l2p(p.y));
    });
    gd.on('plotly_unhover', hide);
    console.info('[Gantt lane guide V4] installed');
  }
  const observer = new MutationObserver(install);
  observer.observe(document.documentElement, {childList:true, subtree:true});
  document.addEventListener('DOMContentLoaded', install);
  install();
})();
