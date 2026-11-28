(function () {
  function setupSidebars() {
    const layout = document.getElementById('quarto-content');
    if (!layout) return;

    const desktop = window.matchMedia('(min-width: 992px)');
    const navbar = document.querySelector('#quarto-header .navbar-container');
    const menu = document.getElementById('navbarCollapse');
    const tools = document.querySelector('#quarto-header .quarto-navbar-tools');
    if (navbar && menu && tools) {
      function placeTools() {
        (desktop.matches ? navbar : menu).appendChild(tools);
      }
      desktop.addEventListener('change', placeTools);
      placeTools();
    }

    const metadata = document.querySelector('.gd-homepage-sidebar');
    const right = document.getElementById('quarto-margin-sidebar');
    if (document.body.classList.contains('gd-homepage') && metadata && right) {
      const original = metadata.parentElement;
      right.classList.remove('zindex-bottom');
      const main = document.getElementById('quarto-document-content');
      function placeMetadata() {
        (desktop.matches ? right : main).appendChild(metadata);
      }
      desktop.addEventListener('change', placeMetadata);
      placeMetadata();
      original.remove();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupSidebars);
  } else {
    setupSidebars();
  }
})();
