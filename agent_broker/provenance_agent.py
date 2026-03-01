"""
Kinexica — Shelf-Life Passport & Provenance Agent
Issues cryptographically signed Shelf-Life Passports (SLPs) for produce,
integrates with SynthiID for watermarking, and prepares Web3 metadata
for smart contract settlement.
"""
# pylint: disable=import-error
import hashlib
import json
import time
from crewai import Agent, Task, LLM


def _generate_slp_hash(payload: dict) -> str:
    """Deterministic SHA-256 hash of the SLP payload for on-chain anchoring."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_provenance_agent(llm: LLM) -> Agent:
    return Agent(
        role="Shelf-Life Passport & Provenance Officer",
        goal=(
            "Issue cryptographically signed Shelf-Life Passports (SLPs) for every "
            "produce batch, integrate SynthiID watermarking for tamper-proof origin "
            "verification, and prepare Web3 metadata for Ethereum smart contract settlement."
        ),
        backstory=(
            "You are a supply chain provenance expert and blockchain architect. "
            "You design Shelf-Life Passport systems that carry the complete digital "
            "provenance of a food item from farm gate to consumer — including PINN-predicted "
            "shelf life, pathogen scan results, temperature history, and fraud clearance. "
            "Your passports are anchored on Ethereum using Kinexica's KinexicaAsset.sol "
            "smart contract, and SynthiID watermarks ensure authenticity."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def get_provenance_task(agent: Agent, batch_payload: dict) -> Task:
    slp = {
        "slp_version":        "2.0",
        "asset_id":           batch_payload.get("asset_id", "UNKNOWN"),
        "commodity":          batch_payload.get("commodity", "produce"),
        "origin_farm":        batch_payload.get("origin_farm", "Unspecified"),
        "origin_district":    batch_payload.get("origin_district", "Unspecified"),
        "harvest_date":       batch_payload.get("harvest_date", "Unknown"),
        "entry_temp_c":       batch_payload.get("entry_temp_c", 0.0),
        "pinn_shelf_life_h":  batch_payload.get("pinn_shelf_life_h", 0.0),
        "pathogen_clear":     batch_payload.get("pathogen_clear", True),
        "fraud_clear":        batch_payload.get("fraud_clear", True),
        "pidr":               batch_payload.get("pidr", 0.0),
        "issued_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issued_by":          "Kinexica Visual-PINN + PINN Engine v3.0",
    }
    slp["slp_hash"] = _generate_slp_hash(slp)

    return Task(
        description=(
            f"Batch SLP payload:\n{json.dumps(slp, indent=2)}\n\n"
            "Tasks:\n"
            "1. Validate all SLP fields for completeness and logical consistency.\n"
            "2. Confirm PINN shelf life is above minimum viable threshold (>24h for retail).\n"
            "3. Confirm pathogen_clear and fraud_clear — if either is False, "
            "   BLOCK the SLP issuance and escalate to Bio-Security / Fraud agents.\n"
            "4. Generate the final SLP document:\n"
            "   - Include slp_hash as the on-chain anchor\n"
            "   - Add QR-code-ready base64 encoding instruction\n"
            "   - Include: 'Scan this QR to verify thermal history on Ethereum testnet'\n"
            "5. Prepare Ethereum smart contract call metadata:\n"
            "   - Function: recordAsset(asset_id, slp_hash, shelf_life, status)\n"
            "   - Gas estimate: ~45,000 wei\n"
            "   - Network: Ethereum Sepolia testnet / Mainnet (production)\n"
            "6. Confirm SynthiID watermark embedding instructions for the produce image.\n"
            "7. Output the final SLP ready for printing on a tamper-evident QR label."
        ),
        expected_output=(
            "Validated Shelf-Life Passport with slp_hash, smart contract call metadata, "
            "SynthiID watermarking instructions, and QR label output."
        ),
        agent=agent,
    )
