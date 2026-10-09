
/* Mobile-only Plotly gesture diagnostic.
   Place in Dash assets/. Remove to revert. */

(function () {
  'use strict';

  if (!('ontouchstart' in window) &&
      !(navigator.maxTouchPoints > 0)) return;

  const selector = '#gantt-chart .js-plotly-plot';
  let lastPlot = null;

  function apply() {
    const plot = document.querySelector(selector);

    if (!plot || !plot._fullLayout || !window.Plotly) return;

    if (plot !== lastPlot ||
        plot._fullLayout.dragmode !== 'zoom') {
      lastPlot = plot;

      window.Plotly.relayout(plot, {
        dragmode: 'zoom'
      }).catch(function () {});
    }
  }

  let pending = false;

  function schedule() {
    if (pending) return;

    pending = true;

    requestAnimationFrame(function () {
      pending = false;
      apply();
    });
  }

  const observer = new MutationObserver(schedule);

  function start() {
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    schedule();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }

  window.addEventListener('orientationchange', function () {
    setTimeout(function () {
      const plot = document.querySelector(selector);

      if (plot && window.Plotly) {
        window.Plotly.Plots.resize(plot);
      }

      schedule();
    }, 350);
  });
})();
