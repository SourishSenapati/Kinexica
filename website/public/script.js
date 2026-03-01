/* ============================================================
   Kinexica Website — script.js
   Interactions: terminal live feed, scroll reveal, nav, form
   ============================================================ */

// ── Scroll-reveal observer ───────────────────────────────────
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        revealObserver.unobserve(e.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(
  '.compare-card, .tech-card, .market-item, .econ-stat, .diagram-card, .formula-block'
).forEach((el) => {
  el.classList.add('reveal');
  revealObserver.observe(el);
});

// ── Sticky nav shadow on scroll ──────────────────────────────
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.style.boxShadow = window.scrollY > 40
    ? '0 4px 32px rgba(0,0,0,0.5)'
    : 'none';
}, { passive: true });

// ── Mobile hamburger ─────────────────────────────────────────
const hamburger = document.getElementById('hamburger');
const navLinks  = document.getElementById('nav-links');
hamburger.addEventListener('click', () => {
  const isOpen = navLinks.classList.toggle('open');
  hamburger.setAttribute('aria-expanded', isOpen);
});
// Close on link click
navLinks.querySelectorAll('a').forEach((a) => {
  a.addEventListener('click', () => {
    navLinks.classList.remove('open');
    hamburger.setAttribute('aria-expanded', false);
  });
});

// ── Live terminal feed ───────────────────────────────────────
const tempEl   = document.getElementById('t-temp');
const ethEl    = document.getElementById('t-eth');
const shelfEl  = document.getElementById('t-shelf');
const statusEl = document.getElementById('t-status');
const pidrEl   = document.getElementById('t-pidr');

function rand(min, max, dp = 2) {
  return (Math.random() * (max - min) + min).toFixed(dp);
}

function updateTerminal() {
  const temp     = parseFloat(rand(9, 22));
  const ethylene = parseFloat(rand(0.4, 3.0));
  const humidity = parseFloat(rand(74, 100));

  // Simplified Arrhenius-inspired shelf life
  const Ea = 50000; const R = 8.314;
  const T  = temp + 273.15;
  const k  = 1e8 * Math.exp(-Ea / (R * T));
  const baseShelf = 200 - ethylene * 15 - Math.max(0, (humidity - 85) * 0.8);
  const shelf = Math.max(10, baseShelf - k * 0.002 * 1000).toFixed(1);

  const pidr   = (k * 0.0001).toFixed(4);
  const stable = parseFloat(shelf) >= 120;

  if (tempEl)   tempEl.textContent   = `${temp.toFixed(1)} °C`;
  if (ethEl)    ethEl.textContent    = `${ethylene.toFixed(2)} ppm`;
  if (shelfEl)  shelfEl.textContent  = `${shelf} hrs`;
  if (pidrEl)   pidrEl.textContent   = pidr;

  if (statusEl) {
    statusEl.textContent  = stable ? 'STABLE' : 'DISTRESSED';
    statusEl.style.color  = stable ? 'var(--green)' : 'var(--red)';
    statusEl.className    = stable ? 't-stable' : 't-red';
  }
}

// Update every 3 seconds
setInterval(updateTerminal, 3000);

// ── Contact form ─────────────────────────────────────────────
const form    = document.getElementById('contact-form');
const success = document.getElementById('form-success');
const submitBtn = document.getElementById('submit-btn');

if (form) {
  form.addEventListener('submit', (e) => {
    e.preventDefault();

    const email = document.getElementById('email').value.trim();
    const name  = document.getElementById('name').value.trim();

    if (!name || !email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      form.querySelectorAll('input[required]').forEach((inp) => {
        if (!inp.value.trim()) inp.style.borderColor = 'var(--red)';
      });
      return;
    }

    // Reset error styles
    form.querySelectorAll('input').forEach((inp) => {
      inp.style.borderColor = '';
    });

    submitBtn.disabled    = true;
    submitBtn.textContent  = 'Sending...';

    // Simulate API submission delay
    setTimeout(() => {
      form.reset();
      submitBtn.style.display   = 'none';
      success.style.display     = 'block';
    }, 900);
  });
}

// ── Smooth active nav highlight ──────────────────────────────
const sections = document.querySelectorAll('section[id]');
const navAs    = document.querySelectorAll('.nav-links a');

const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        const id = e.target.id;
        navAs.forEach((a) => {
          a.style.color = a.getAttribute('href') === `#${id}`
            ? 'var(--white)'
            : '';
        });
      }
    });
  },
  { threshold: 0.4 }
);
sections.forEach((s) => sectionObserver.observe(s));
