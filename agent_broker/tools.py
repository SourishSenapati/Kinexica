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


@tool("Mint Immutable Smart Contract")
def mint_smart_contract(asset_id: str, temp: float, ethylene: float, hours: float, price: float) -> str:
    """
    Deploys a cryptographically secure proof-of-kinetics verification smart contract 
    to the local offline Ethereum node (Ganache/Eth-Tester).
    You MUST provide exactly: asset_id, temperature, ethylene ppm, remaining hours, and the negotiated liquidated price.
    """
    print(
        f"\n[BLOCKCHAIN] Deploying Proof-of-Kinetics for {asset_id} at ${price}...")

    h, b = deploy_kinexica_contract(asset_id, float(
        temp), float(ethylene), float(hours), float(price))

    # Update SQLite Database so UI is synced
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "inventory.db")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        "UPDATE assets SET status='Liquidated', tx_hash=?, block_number=? WHERE asset_id=?", (h, b, asset_id))
    conn.commit()
    conn.close()

    return f"Smart contract minted securely. TxHash: {h} Block: {b}"
