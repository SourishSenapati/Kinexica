# Kinexica — Autonomous AI at the Food Spoilage Frontier

> **"Instead of watching losses grow every year, Kinexica converts decay into compounding revenue."**

## Live Website

Deployed on Vercel: [kinexica.vercel.app](https://kinexica.vercel.app)

---

## What Is Kinexica?

Kinexica is the world's first autonomous end-to-end food spoilage prevention platform combining:

| Layer                | Technology                                        | Output                     |
| -------------------- | ------------------------------------------------- | -------------------------- |
| **Sensor Telemetry** | IoT edge (Temp · Humidity · Ethylene)             | Real-time PIDR stream      |
| **PINN Engine**      | Arrhenius physics + neural inference (5→128→64→2) | Shelf-life forecast ± σ    |
| **Vision (LENS)**    | OpenCV Laplacian diffusion analysis               | Pathogen / fraud detection |
| **AI Broker**        | CrewAI agent swarm                                | Autonomous liquidation     |
| **Blockchain**       | Web3 smart contracts + SynthiID                   | On-chain audit trail       |

---

## Website Diagrams (Python-generated)

All diagrams built with **matplotlib** — no AI image generation:

| File                     | Contents                                                  |
| ------------------------ | --------------------------------------------------------- |
| `economic_analysis.png`  | 10-year cumulative financial impact bar chart             |
| `capability_radar.png`   | Kinexica vs Traditional vs IoT sensors radar              |
| `arrhenius_decay.png`    | Physics-driven shelf-life decay at 4 temperature profiles |
| `market_opportunity.png` | $2.4T TAM donut breakdown by sector                       |
| `architecture.png`       | End-to-end PINN pipeline architecture flow                |

Regenerate diagrams:

```bash
python website/generate_diagrams.py
```

---

## Local Development

```bash
# Serve the website locally (any static server)
cd website/public
python -m http.server 3000
# → http://localhost:3000
```

---

## Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# From the website folder
cd website
vercel --prod
```

---

## PINN Engine (Backend)

```bash
# Install dependencies
pip install -r requirements.txt

# Train the PINN model (500 epochs)
python -m pinn_engine.fast_train

# Start FastAPI backend
python main.py

# Start Flet dashboard
python -m dashboard.app
```
