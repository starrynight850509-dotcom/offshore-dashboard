/* F2 Gantt lane guide: annotation labels and Plotly bars, Dash-safe. */
(function () {
  'use strict';
  const ROOT_ID = 'gantt-chart';
  const FLAG = '__ganttLaneGuideV3';

  function install() {
    const root = document.getElementById(ROOT_ID);
    const gd = root && root.querySelector('.js-plotly-plot');
    if (!gd || gd[FLAG] || !gd._fullLayout || typeof gd.on !== 'function') return;
    gd[FLAG] = true;
    gd.style.position = 'relative';

    // Remove guide nodes from older versions if present.
    gd.querySelectorAll('.gantt-lane-hover-guide').forEach(el => el.remove());
    const line = document.createElement('div');
    line.className = 'gantt-lane-hover-guide';
    Object.assign(line.style, {
      position: 'absolute', height: '0px',
      borderTop: '1px solid rgba(100, 116, 139, 0.25)',
      pointerEvents: 'none', zIndex: '1',
      visibility: 'visible', opacity: '1', display: 'none'
    });
    gd.appendChild(line);

    const hide = () => { line.style.display = 'none'; };
    function bounds() {
      const l = gd._fullLayout;
      const x = l && l.xaxis;
      const y = l && l.yaxis; // Gantt; yaxis2 is the wave forecast.
      if (!x || !y) return null;
      return {left:x._offset, width:x._length,
        top:y._offset, bottom:y._offset+y._length, y};
    }
    function draw(clientY) {
      const b = bounds();
      if (!b) return hide();
      const top = clientY - gd.getBoundingClientRect().top;
      if (top < b.top - 3 || top > b.bottom + 3) return hide();
      Object.assign(line.style, {
        left: b.left + 'px', width: b.width + 'px',
        top: top + 'px', display: 'block'
      });
    }
    function annotationCenter(target) {
      const a = target.closest('.annotation');
      if (!a) return null;
      // Only annotations belonging to the Gantt y-axis, not dates/wave notes.
      const data = gd._fullLayout && gd._fullLayout.annotations;
      if (!data) return null;
      const annotations = Array.from(gd.querySelectorAll('.annotation'));
      const index = annotations.indexOf(a);
      if (index < 0 || !data[index] || data[index].yref !== 'y') return null;
      const r = a.getBoundingClientRect();
      return r.top + r.height / 2;
    }
    function move(e) {
      const t = e.target;
      if (!(t instanceof Element)) return hide();
      const a = annotationCenter(t);
      if (a !== null) return draw(a);
      // The plot drag area can be outside .cartesianlayer.
      if (t.closest('.nsewdrag, .draglayer, .cartesianlayer, .plot')) return draw(e.clientY);
      hide();
    }
    gd.addEventListener('pointermove', move);
    gd.addEventListener('pointerleave', hide);
    gd.on('plotly_hover', function (evt) {
      const p = evt && evt.points && evt.points[0];
      const b = bounds();
      if (!p || !b || p.yaxis !== b.y || p.y == null) return;
      draw(gd.getBoundingClientRect().top + b.y._offset + b.y.l2p(p.y));
    });
    gd.on('plotly_unhover', hide);
    console.info('[Gantt lane guide V3] installed');
  }

  const observer = new MutationObserver(install);
  observer.observe(document.documentElement, {childList:true, subtree:true});
  document.addEventListener('DOMContentLoaded', install);
  install();
})();
