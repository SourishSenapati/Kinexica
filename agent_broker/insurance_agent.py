"""
Kinexica — Parametric Insurance Oracle Agent
Monitors PIDR breach events, calculates autonomous insurance payouts
using Arrhenius-derived loss functions, and triggers smart contract
settlement on the KinexicaAsset.sol Ethereum contract.
"""
# pylint: disable=import-error
from crewai import Agent, Task, LLM


def get_insurance_agent(llm: LLM) -> Agent:
    return Agent(
        role="Parametric Insurance Oracle",
        goal=(
            "Evaluate PIDR breach events against insured shelf-life thresholds, "
            "calculate the precise insurance payout using the physics-derived loss function, "
            "and trigger autonomous smart contract settlement without human intervention."
        ),
        backstory=(
            "You are a parametric insurance actuary and smart contract oracle specialist. "
            "Unlike traditional insurers, you need no loss adjusters or field inspections — "
            "the PINN-derived PIDR (Physics-Informed Decay Rate) is the single source of truth "
            "for triggering payouts. You calculate payouts using a time-decay liquidation curve: "
            "  > 72h remaining: Base payout (20% of insured value)\n"
            "  24-72h remaining: Standard payout (50% of insured value)\n"
            "  < 24h remaining: Full emergency payout (90% of insured value)\n"
            "You draft the Ethereum transaction payload for KinexicaAsset.sol's "
            "triggerPayout(asset_id, payout_amount) function."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def get_insurance_task(agent: Agent, payload: dict) -> Task:
    asset_id = payload.get("asset_id",              "UNKNOWN")
    shelf_h = payload.get("estimated_hours_remaining", 0.0)
    insured_value = payload.get("insured_value_usd",     1000.0)
    pidr = payload.get("pidr",                   0.0)
    temp = payload.get("current_temp_c",          20.0)
    ethylene = payload.get("peak_ethylene_ppm",        0.0)

    # Payout tier calculation
    if shelf_h < 24:
        tier, pct = "TIER-1 EMERGENCY", 0.90
    elif shelf_h < 72:
        tier, pct = "TIER-2 STANDARD",  0.50
    else:
        tier, pct = "TIER-3 BASE",      0.20
    payout = round(insured_value * pct, 2)

    return Task(
        description=(
            f"PARAMETRIC INSURANCE EVENT\n"
            f"Asset ID         : {asset_id}\n"
            f"PIDR             : {pidr:.6f}\n"
            f"Temp             : {temp}°C\n"
            f"Ethylene         : {ethylene} ppm\n"
            f"Hours Remaining  : {shelf_h:.1f} h\n"
            f"Insured Value    : ${insured_value:,.2f} USD\n"
            f"Payout Tier      : {tier}\n"
            f"Payout Amount    : ${payout:,.2f} USD  ({pct:.0%})\n\n"
            "Tasks:\n"
            "1. Confirm the PIDR breach is genuine (PIDR > 0.005 = breach threshold).\n"
            "2. Validate payout tier against insured shelf-life SLA in the policy.\n"
            "3. Draft the Ethereum transaction:\n"
            "   - Contract: KinexicaAsset.sol\n"
            "   - Function: triggerPayout(string asset_id, uint payout_wei)\n"
            "   - Convert USD payout to ETH at current oracle price.\n"
            "   - Include gas limit: 80,000 wei\n"
            "4. Generate a one-page Parametric Claim Settlement Summary for the insured party.\n"
            "5. Log event to IPFS for immutable audit trail."
        ),
        expected_output=(
            "Payout validation summary, Ethereum transaction payload, "
            "and parametric claim settlement document."
        ),
        agent=agent,
    )
