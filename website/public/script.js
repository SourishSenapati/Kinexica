/* ============================================================
   Kinexica Website — script.js
   Interactive: terminal live feed, pricing, scroll reveal, nav, form
   ============================================================ */

'use strict';

// ── USD prices (monthly) ────────────────────────────────────
const PRICES = {
  starter:    { usd: 199,  label: 'Starter B2B' },
  pro:        { usd: 599,  label: 'Professional' },
  enterprise: { usd: null, label: 'Enterprise B2G' },   // null = "Custom"
};
const ANNUAL_DISCOUNT = 0.80;   // 20% off
let currentCurrency = 'usd';
let isAnnual        = false;
let usdToInr        = 84.50;   // fallback; updated from live API

// ── Format price ───────────────────────────────────────────
function formatPrice(usd) {
  if (usd === null) return ['Custom', ''];
  let amount = usd;
  if (isAnnual) amount = Math.round(usd * ANNUAL_DISCOUNT);
  if (currentCurrency === 'inr') {
    const inr = Math.round(amount * usdToInr);
    return [inr.toLocaleString('en-IN'), '/mo'];
  }
  return [amount.toLocaleString('en-US'), '/mo'];
}

function symFor(currency) {
  return currency === 'inr' ? '₹' : '$';
}

function annualTotal(usd) {
  if (usd === null) return '';
  const monthly = Math.round(usd * ANNUAL_DISCOUNT);
  const yearly  = monthly * 12;
  if (currentCurrency === 'inr') {
    return `₹${Math.round(yearly * usdToInr).toLocaleString('en-IN')}`;
  }
  return `$${yearly.toLocaleString('en-US')}`;
}

// ── Render prices ──────────────────────────────────────────
function renderPrices() {
  const sym = symFor(currentCurrency);

  Object.entries(PRICES).forEach(([key, data]) => {
    const amtEl  = document.getElementById(`price-${key}`);
    const symEl  = document.getElementById(`sym-${key}`);
    const annEl  = document.getElementById(`annual-${key}`);
    const annTot = document.getElementById(`annual-${key}-total`);

    if (!amtEl) return;

    const [amount, period] = formatPrice(data.usd);
    symEl.textContent = data.usd === null ? '' : sym;
    amtEl.textContent = amount;

    if (annEl) {
      if (isAnnual && data.usd !== null) {
        annEl.classList.remove('hidden');
        if (annTot) annTot.textContent = annualTotal(data.usd);
      } else {
        annEl.classList.add('hidden');
      }
    }
  });
}

// ── Fetch live INR rate ────────────────────────────────────
async function fetchINRRate() {
  try {
    const res  = await fetch('https://open.er-api.com/v6/latest/USD');
    const data = await res.json();
    if (data && data.rates && data.rates.INR) {
      usdToInr = data.rates.INR;
      const rateEl = document.getElementById('rate-value');
      if (rateEl) {
        rateEl.textContent = `1 USD = ₹${usdToInr.toFixed(2)}`;
      }
    }
  } catch {
    // silently fall back to hardcoded rate
  }
}

// ── Currency toggle ────────────────────────────────────────
const btnUSD = document.getElementById('btn-usd');
const btnINR = document.getElementById('btn-inr');
const rateNote = document.getElementById('rate-note');

function setCurrency(currency) {
  currentCurrency = currency;
  btnUSD.classList.toggle('active', currency === 'usd');
  btnINR.classList.toggle('active', currency === 'inr');
  btnUSD.setAttribute('aria-pressed', currency === 'usd');
  btnINR.setAttribute('aria-pressed', currency === 'inr');

  if (rateNote) {
    if (currency === 'inr') {
      rateNote.classList.remove('hidden');
      fetchINRRate();
    } else {
      rateNote.classList.add('hidden');
    }
  }
  renderPrices();
}

if (btnUSD) btnUSD.addEventListener('click', () => setCurrency('usd'));
if (btnINR) btnINR.addEventListener('click', () => setCurrency('inr'));

// ── Billing toggle ─────────────────────────────────────────
const billingSwitch = document.getElementById('billing-switch');
if (billingSwitch) {
  billingSwitch.addEventListener('change', () => {
    isAnnual = billingSwitch.checked;
    renderPrices();
  });
}

// ── Initial render ─────────────────────────────────────────
renderPrices();

// ── Scroll-reveal observer ─────────────────────────────────
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        revealObserver.unobserve(e.target);
      }
    });
  },
  { threshold: 0.10 }
);

document.querySelectorAll(
  '.compare-card, .tech-card, .market-item, .econ-stat, ' +
  '.diagram-card, .formula-block, .pricing-card'
).forEach((el) => {
  el.classList.add('reveal');
  revealObserver.observe(el);
});

// ── Sticky nav shadow on scroll ────────────────────────────
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.style.boxShadow = window.scrollY > 40
    ? '0 4px 32px rgba(0,0,0,0.5)'
    : 'none';
}, { passive: true });

// ── Mobile hamburger ───────────────────────────────────────
const hamburger = document.getElementById('hamburger');
const navLinks  = document.getElementById('nav-links');

hamburger.addEventListener('click', () => {
  const isOpen = navLinks.classList.toggle('open');
  hamburger.setAttribute('aria-expanded', String(isOpen));
});
// Close on link click
navLinks.querySelectorAll('a').forEach((a) => {
  a.addEventListener('click', () => {
    navLinks.classList.remove('open');
    hamburger.setAttribute('aria-expanded', 'false');
  });
});

// ── Live terminal feed (Arrhenius-inspired) ─────────────────
const tempEl   = document.getElementById('t-temp');
const ethEl    = document.getElementById('t-eth');
const shelfEl  = document.getElementById('t-shelf');
const statusEl = document.getElementById('t-status');
const pidrEl   = document.getElementById('t-pidr');

function rand(min, max, dp = 2) {
  return (Math.random() * (max - min) + min).toFixed(dp);
}

function updateTerminal() {
  const temp      = parseFloat(rand(9, 28));
  const ethylene  = parseFloat(rand(0.4, 3.5));
  const humidity  = parseFloat(rand(70, 100));
  const EA = 50000; const R = 8.314;
  const Tk = temp + 273.15;
  const k  = 1e8 * Math.exp(-EA / (R * Tk));
  const base  = 200 - ethylene * 15 - Math.max(0, (humidity - 85) * 0.8);
  const shelf = Math.max(10, base - k * 0.002 * 1000).toFixed(1);
  const pidr  = (k * 0.0001).toFixed(4);
  const stable = parseFloat(shelf) >= 120;

  if (tempEl)   tempEl.textContent  = `${temp.toFixed(1)} °C`;
  if (ethEl)    ethEl.textContent   = `${ethylene.toFixed(2)} ppm`;
  if (shelfEl)  shelfEl.textContent = `${shelf} hrs`;
  if (pidrEl)   pidrEl.textContent  = pidr;
  if (statusEl) {
    statusEl.textContent = stable ? 'STABLE' : 'DISTRESSED';
    statusEl.style.color = stable ? 'var(--green)' : 'var(--red)';
  }
}
setInterval(updateTerminal, 3000);

// ── Contact form ───────────────────────────────────────────
const form      = document.getElementById('contact-form');
const success   = document.getElementById('form-success');
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
    form.querySelectorAll('input').forEach((inp) => { inp.style.borderColor = ''; });
    submitBtn.disabled    = true;
    submitBtn.textContent = 'Sending...';
    setTimeout(() => {
      form.reset();
      submitBtn.classList.add('hidden');
      success.classList.remove('hidden');
    }, 900);
  });
}

// ── Active nav highlight on scroll ─────────────────────────
const sections = document.querySelectorAll('section[id]');
const navAs    = document.querySelectorAll('.nav-links a');

const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        const id = e.target.id;
        navAs.forEach((a) => {
          a.style.color = (a.getAttribute('href') === `#${id}`) ? 'var(--white)' : '';
        });
      }
    });
  },
  { threshold: 0.4 }
);
sections.forEach((s) => sectionObserver.observe(s));

// ── Dynamic Canvas Charts (Chart.js) ─────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#a3a3a3';
  Chart.defaults.font.family = '"Inter", sans-serif';

  // 1. Radar Chart
  const radarCtx = document.getElementById('capabilityRadarChart');
  if (radarCtx) {
    new Chart(radarCtx, {
      type: 'radar',
      data: {
        labels: ['Predictive AI', 'Detection Speed', 'Hardware Cost', 'Classification', 'Integration', 'Scalability'],
        datasets: [
          {
            label: 'Kinexica',
            data: [95, 100, 90, 85, 90, 95],
            backgroundColor: 'rgba(16, 185, 129, 0.2)',
            borderColor: '#10b981',
            pointBackgroundColor: '#10b981'
          },
          {
            label: 'Basic IoT Sensors',
            data: [30, 80, 50, 20, 60, 70],
            backgroundColor: 'rgba(59, 130, 246, 0.2)',
            borderColor: '#3b82f6',
            pointBackgroundColor: '#3b82f6'
          },
          {
            label: 'Traditional Cold-Chain',
            data: [10, 20, 80, 10, 30, 40],
            backgroundColor: 'rgba(239, 68, 68, 0.2)',
            borderColor: '#ef4444',
            pointBackgroundColor: '#ef4444'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            pointLabels: { color: '#a3a3a3', font: { size: 10 } },
            ticks: { display: false, max: 100, min: 0 }
          }
        },
        plugins: {
          legend: { position: 'bottom', labels: { color: '#ffffff' } }
        }
      }
    });
  }

  // 2. Arrhenius Decay Line Chart
  const decayCtx = document.getElementById('arrheniusDecayChart');
  if (decayCtx) {
    const timeLabels = Array.from({length: 15}, (_, i) => `${i * 12}h`);
    // Simulated realistic curve drops
    const temp5 = timeLabels.map((_, i) => Math.max(0, 100 - (Math.pow(1.08, i) * 2)));
    const temp15 = timeLabels.map((_, i) => Math.max(0, 100 - (Math.pow(1.18, i) * 3)));
    const temp25 = timeLabels.map((_, i) => Math.max(0, 100 - (Math.pow(1.35, i) * 4)));
    const temp35 = timeLabels.map((_, i) => Math.max(0, 100 - (Math.pow(1.6, i) * 5)));
    
    new Chart(decayCtx, {
      type: 'line',
      data: {
        labels: timeLabels,
        datasets: [
          { label: '5°C (Stable)', data: temp5, borderColor: '#3b82f6', tension: 0.4 },
          { label: '15°C', data: temp15, borderColor: '#10b981', tension: 0.4 },
          { label: '25°C', data: temp25, borderColor: '#f59e0b', tension: 0.4 },
          { label: '35°C (Critical)', data: temp35, borderColor: '#ef4444', tension: 0.4 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { 
            title: { display: true, text: 'Quality Index (%)', color: '#a3a3a3' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            min: 0, max: 100
          },
          x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
        },
        plugins: {
          legend: { labels: { color: '#ffffff' } }
        }
      }
    });
  }

  // 3. Economic Analysis Bar Chart
  const econCtx = document.getElementById('economicChart');
  if (econCtx) {
    new Chart(econCtx, {
      type: 'bar',
      data: {
        labels: ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5', 'Year 6', 'Year 7', 'Year 8', 'Year 9', 'Year 10'],
        datasets: [
          {
            label: 'Kinexica Profit Projection',
            data: [-50, -20, 15, 60, 120, 190, 260, 340, 430, 535],
            backgroundColor: '#10b981'
          },
          {
            label: 'Traditional Cumulative Loss',
            data: [-20, -42, -65, -88, -112, -135, -158, -181, -202, -220],
            backgroundColor: '#ef4444'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { 
            title: { display: true, text: 'Financial Impact (USD Thousands)', color: '#a3a3a3' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' } 
          },
          x: { grid: { display: false } }
        },
        plugins: {
          legend: { labels: { color: '#ffffff' } }
        }
      }
    });
  }

  // 4. Market Doughnut Chart
  const marketCtx = document.getElementById('marketChart');
  if (marketCtx) {
    new Chart(marketCtx, {
      type: 'doughnut',
      data: {
        labels: ['B2B Warehouses (%)', 'B2C Retail (%)', 'B2G Govt (%)', 'Aquaculture (%)', 'Carbon Markets (%)'],
        datasets: [{
          data: [40, 22, 18, 12, 8],
          backgroundColor: ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#06b6d4'],
          borderWidth: 0,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '70%',
        plugins: {
          legend: { position: 'right', labels: { color: '#ffffff', font: { size: 12 } } }
        }
      }
    });
  }
}
