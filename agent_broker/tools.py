"""
Tools for the agent broker.
"""

import os
from crewai.tools import tool


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
