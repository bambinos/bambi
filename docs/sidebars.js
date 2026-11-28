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

    const toc = document.getElementById('TOC');
    if (toc) {
      let selectedLink = null;
      function selectLink(link) {
        toc.querySelectorAll('a.nav-link').forEach(function (item) {
          item.classList.toggle('active', item === link);
        });
      }

      // Near the page bottom, scrolling may stop before the selected heading reaches the top.
      toc.addEventListener('click', function (event) {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        const link = event.target.closest('a[data-scroll-target]');
        if (!link) return;
        const section = document.getElementById(decodeURIComponent(
          link.getAttribute('data-scroll-target').slice(1)
        ));
        const offset = parseFloat(getComputedStyle(section).scrollMarginTop);
        const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
        const destination = Math.max(0, Math.min(
          window.scrollY + section.getBoundingClientRect().top - offset, maxScroll
        ));
        selectedLink = Math.abs(destination - window.scrollY) > 1 ? link : null;
        selectLink(link);
      });

      document.addEventListener('scrollend', function () {
        if (!selectedLink) return;
        const link = selectedLink;
        selectedLink = null;
        requestAnimationFrame(function () { selectLink(link); });
      });
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
