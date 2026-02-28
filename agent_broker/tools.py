# pylint: disable=import-error, line-too-long, broad-except, consider-using-f-string, import-outside-toplevel
"""
Tools for the agent broker.
"""

import os
import sqlite3
from crewai.tools import tool
from blockchain.deploy import deploy_kinexica_contract


@tool("Save Smart Contract to Ledger")
def save_contract_to_ledger(contract_text: str) -> str:
    """
    Saves the final generated liquidation smart contract to the local 
    inventory ledger (a text file).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger_path = os.path.join(base_dir, "data", "smart_contracts_ledger.txt")

    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write("\n--- NEW SMART CONTRACT ---\n")
        f.write(contract_text)
        f.write("\n--------------------------\n")

    return f"Smart contract secured and written to {ledger_path}"


@tool("Broadcast Webhook to Supply Chain")
def broadcast_webhook(payload_summary: str) -> str:
    """
    Simulates broadcasting an emergency liquidation webhook to pending buyers 
    and logistics operators.
    """
    print(f"\n[NETWORK] Broadcasting Webhook: {payload_summary}")
    return "Webhook broadcast successful. Stakeholders notified."


# --- Advanced Bio-Manufacturing Niche (B2B) ---
@tool("Calculate Biomass Carbon Yield")
def calculate_biomass_carbon_yield(mass_kg: float, rot_percentage: float) -> str:
    """
    Calculates the exact accessible glucose and carbon substrate available 
    for Precision Fermentation matching instead of base composting.
    """
    glucose_yield_kg = mass_kg * rot_percentage * 0.45
    print(
        f"\n[BIOTECH-MATCH] Calculated {glucose_yield_kg:.2f} kg of fermentable glucose.")
    return f"Available substrate: {glucose_yield_kg:.2f} kg pure fermentable carbon. Ready for Biotech matching."


# --- Micro-Logistics & Gig Economy Niche (B2B2C) ---
@tool("Dispatch Gig Driver (Kinexica Rescue API)")
def dispatch_last_mile_gig_driver(asset_location: str, destination_lab: str, mass_kg: float) -> str:
    """
    Pings the 'Kinexica Rescue API' to dispatch a local gig-economy driver 
    (Uber Freight style) to move distressed biomass immediately.
    """
    fee = 10.0 + (mass_kg * 0.05)  # Base fee + weight variable
    print(
        f"\n[GIG-RESCUE] Driver dispatched from {asset_location} to {destination_lab}. Take Rate fee: ${fee:.2f}")
    return f"Gig driver dispatched successfully. Platform revenue generated: ${fee:.2f}"


# --- Dynamic Retail Pricing Niche (B2B2C) ---
@tool("Calculate Dynamic Shelf Price")
def calculate_dynamic_shelf_price(base_price: float, remaining_hours: float) -> str:
    """
    Calculates instantaneous markdown for electronic shelf labels based on 
    the PINN reaction-diffusion remaining lifespan.
    """
    if remaining_hours <= 18 and remaining_hours > 0:
        discount = 0.30  # 30% off
    elif remaining_hours <= 6 and remaining_hours > 0:
        discount = 0.75  # 75% off
    else:
        discount = 0.0

    new_price = base_price * (1 - discount)
    print(
        f"\n[DYNAMIC-FRONTLINE] Remaining: {remaining_hours}h. Markdown ({discount*100}%). New Price: ${new_price:.2f}")
    return f"Electronic Shelf Label updated to ${new_price:.2f}"


@tool("Mint Immutable Smart Contract")
def mint_smart_contract(asset_id: str, temp: float, ethylene: float, hours: float, price: float, mass_kg: float = 1000.0) -> str:
    """
    Deploys a cryptographically secure proof-of-kinetics verification 
    smart contract to the local offline Ethereum node (Ganache/Eth-Tester).
    You MUST provide exactly: asset_id, temperature, ethylene ppm, 
    remaining hours, and the negotiated liquidated price.
    Also tracks carbon credit MRV offset via ERC-20 KCT issuance.
    """
    print(
        f"\n[BLOCKCHAIN] Deploying Proof-of-Kinetics for {asset_id} at ${price}...")

    # MRV Carbon Credit Protocol (Phase 2)
    ef_landfill = 0.5  # kg CO2e per kg in landfill
    ef_upcycle = 0.1   # kg CO2e per kg in bio-feed upcycle

    co2e_saved_kg = mass_kg * (ef_landfill - ef_upcycle)
    grams_methane_prevented = int((co2e_saved_kg / 25.0) * 1000)

    h, b = deploy_kinexica_contract(asset_id, float(
        temp), float(ethylene), float(hours), float(price), grams_methane_prevented)

    # Update SQLite Database so UI is synced
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "inventory.db")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        "UPDATE assets SET status='Liquidated', tx_hash=?, block_number=? WHERE asset_id=?", (h, b, asset_id))
    conn.commit()
    conn.close()

    import json
    mrv_report = json.dumps({
        "asset_id": asset_id,
        "mass_rescued_kg": mass_kg,
        "location_offset": "Anonymized via Edge Truncation",
        "co2e_saved_kg": co2e_saved_kg,
        "kct_tokens_minted": grams_methane_prevented,
        "blockchain_tx_receipt": h
    }, indent=2)

    return (f"Smart contract minted securely. TxHash: {h} Block: {b}\n"
            f"MRV Report Generated:\n{mrv_report}")
