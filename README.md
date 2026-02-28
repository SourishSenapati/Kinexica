<div align="center">
  <h1>🌿 Kinexica SpoilSense-Broker</h1>
  <p>An Autonomous Deep-Learning Supply Chain & Parametric Insurance Oracle Network</p>
</div>

---

## 🚀 The Vision

SpoilSense-Broker is a state-of-the-art **B2B / B2G / B2C** biological evaluation and liquidation system for global supply chains. When perishable goods (e.g., agriculture, medical supplies, biomass) degrade in transit, this system acts as the absolute ultimate arbiter of truth.

It utilizes three synchronized verticals:

1. **Edge Telemetry:** Ingesting GPS-verified environmental data alongside continuous computer-vision pathology scans (analyzing surface variance, Laplacian blur, and specific chemical adulteration signatures like Calcium Carbide).
2. **Physics-Informed Neural Networks (PINN):** Implementing real-time 5-dimensional tensors representing the Arrhenius Decay Kinetics curve to scientifically measure, predict, and log the exact remaining shelf life down to the second.
3. **Multi-Agent Swarm Arbitration:** Triggering decentralized Llama3 endpoints to automatically dispatch gig drivers, liquidate distressed biomass, and execute final payment transfers via localized Ganache/Parametric Smart Contracts immediately upon trigger condition.

## 🏗️ Architecture Stack

- **Central Orchestrator:** FastAPI (Asynchronous, High-Concurrency Request Routing)
- **Mathematical Baseline:** PyTorch PINN (Arrhenius Chemical Kinetics) + OpenCV
- **Blockchain Oracle Logic:** Web3.py + Solidity (Ganache Parametric Smart Contract Network)
- **Agentic Negotiators:** Swarm LLMs via Local Ollama Infrastructure
- **Cross-Platform Mobile Client:** Flet (Progressive Web Application + Native Android APK Wrapper)
- **Database Backend:** SQLite concurrency thread-locking model

---

## 📂 Core Verticals

| Directory        | Purpose                     | Detail                                                                                                                                                                |
| :--------------- | :-------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_pipeline/` | Edge sensor ingestion       | Simulates/Ingests JSON blobs from Warehouse/Transit IoT Arduino nodes.                                                                                                |
| `pinn_engine/`   | Chemical thermodynamics     | 5D PyTorch PINN `[Temp, Humidity, Ethylene, CV_Variance, CV_Intensity]` calculating explicit Arrhenius biological decay and detecting chemical ripening adulteration. |
| `blockchain/`    | Parametric Oracle Contracts | Solidity smart contracts dictating fractional payout percentages based on precise PINN deterioration logs and time-of-loss.                                           |
| `agent_broker/`  | Multi-agent negotiation     | Ollama LLM Swarm infrastructure mapping function-calls to tools like Carbon Yield Calculations, Dynamic Shelf Pricing, & Gig Dispatch routing.                        |
| `dashboard/`     | B2B & B2C Edge Client       | Responsive Mobile App UI handling PWA and Native-Compile Android integration exposing localized scan data directly from FastAPI backend across Localtunnel.           |
| `main.py`        | API Aggregator              | Asynchronous master bootloader executing continuous surveillance loops.                                                                                               |

---

## ⚙️ Usage & Pipeline Deployment

To run the Kinexica architecture identically within your stack, ensure the following systems are live:

### 1. Boot the Network Oracle (Web3)

Spin up your local Ethereum/Ganache GUI network on `127.0.0.1:7545`. Load the `KinexicaAsset.sol` via Remix IDE and inject the deployed contract address string into the root of `main.py` mapping.

### 2. Trigger Ollama Background Daemon

Ensure Llama3 is running in memory inside your system tray bindings for API port mapping (`11434`) matching the FastAPI Agent.

### 3. Boot The Core

Open your root directory and spin up the ASGI endpoint server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Penetrate the Local Firewall

Use `npx localtunnel` to tunnel traffic from Port `8000` to the internet. Expose the URL dynamically into `dashboard/app.py` for client binding.

### 5. Deploy Mobile UI

Compile your localized UI into an executable Android binary via:

```bash
flet build apk dashboard --module-name app
```

Alternatively, bypass the SDK compile entirely with a direct web fallback by spinning Flet on Port `8502` and passing it through a Progressive Web App (PWA) tunnel.

---

## 🛡️ Trust Protocols & Data Assurance

The absolute value of Kinexica is trust.

- You cannot spoof the Arrhenius physics calculations.
- You cannot revert the Parametric Insurtech Payouts on the blockchain.
- By merging un-spoofable mathematical decay models with localized supply chain validation, Kinexica eliminates 100% of human subjectivity during the liquidation process of degrading biological assets.

Developed by **Sourish Senapati**.
