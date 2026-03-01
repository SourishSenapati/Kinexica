"""
Kinexica — Master Agent Orchestrator
════════════════════════════════════
Intelligently routes spoilage signals, pathogen detections, fraud alerts,
and distress events to the appropriate specialized CrewAI agent swarms.

Agents available:
  1. Kinetics Assessor     — evaluates thermodynamic decay severity
  2. Autonomous Broker     — maximizes distressed asset value recovery
  3. Gig Routing Agent     — dispatches last-mile drivers
  4. Carbon MRV Agent      — calculates and certifies carbon credits
  5. Bio-Security Agent    — maps pathogen outbreaks for B2G dashboards
  6. Fraud Investigator    — builds enforcement packages for fraud cases
  7. Provenance Officer    — issues Shelf-Life Passports on-chain
  8. Insurance Oracle      — triggers parametric payouts via smart contracts

Usage:
  from agent_broker.orchestrator import handle_event
  handle_event({"event_type": "distress",    ...})
  handle_event({"event_type": "pathogen",    ...})
  handle_event({"event_type": "fraud",       ...})
  handle_event({"event_type": "provenance",  ...})
  handle_event({"event_type": "insurance",   ...})
  handle_event({"event_type": "carbon",      ...})
"""
# pylint: disable=import-error

from crewai import Crew, Process, LLM

from agent_broker.negotiation_agent import (
    trigger_negotiation_swarm,
)
from agent_broker.carbon_agent import (
    get_carbon_agent, get_carbon_task,
)
from agent_broker.biosecurity_agent import (
    get_biosecurity_agent, get_biosecurity_task,
)
from agent_broker.fraud_agent import (
    get_fraud_agent, get_fraud_task,
)
from agent_broker.provenance_agent import (
    get_provenance_agent, get_provenance_task,
)
from agent_broker.insurance_agent import (
    get_insurance_agent, get_insurance_task,
)
from agent_broker.routing_agent import (
    get_routing_agent, get_routing_task,
)


# ── LLM Singleton (local Ollama — zero cloud cost) ─────────────────────────
def _llm() -> LLM:
    return LLM(model="ollama/llama3")


# ── Event Handlers ─────────────────────────────────────────────────────────

def handle_distress(payload: dict) -> str:
    """
    Handles a Distressed asset event.
    Triggers: Kinetics Assessor → Autonomous Broker → Gig Routing Agent.
    """
    print("\n[ORCHESTRATOR] Routing DISTRESS event → Negotiation + Routing swarm")
    return str(trigger_negotiation_swarm(payload))


def handle_pathogen(detections: list) -> str:
    """
    Handles pathogen detection batch.
    Triggers: Bio-Security Agent → issues government alerts.
    """
    print("\n[ORCHESTRATOR] Routing PATHOGEN event → Bio-Security swarm")
    llm = _llm()
    agent = get_biosecurity_agent(llm)
    task = get_biosecurity_task(agent, detections)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    print("\n[BIO-SECURITY ALERT]\n", result)
    return str(result)


def handle_fraud(fraud_result: dict) -> str:
    """
    Handles a fraud detection from Visual-PINN.
    Triggers: Fraud Investigator → builds enforcement package.
    """
    print("\n[ORCHESTRATOR] Routing FRAUD event → Fraud Investigator swarm")
    llm = _llm()
    agent = get_fraud_agent(llm)
    task = get_fraud_task(agent, fraud_result)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    print("\n[FRAUD ENFORCEMENT PACKAGE]\n", result)
    return str(result)


def handle_provenance(batch_payload: dict) -> str:
    """
    Issues a Shelf-Life Passport for a cleared batch.
    Triggers: Provenance Officer → validates + stamps SLP on-chain.
    """
    print("\n[ORCHESTRATOR] Routing PROVENANCE event → Provenance Officer swarm")
    llm = _llm()
    agent = get_provenance_agent(llm)
    task = get_provenance_task(agent, batch_payload)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    print("\n[SHELF-LIFE PASSPORT]\n", result)
    return str(result)


def handle_insurance(payload: dict) -> str:
    """
    Processes a parametric insurance PIDR breach event.
    Triggers: Insurance Oracle → calculates payout + drafts tx.
    """
    print("\n[ORCHESTRATOR] Routing INSURANCE event → Parametric Oracle swarm")
    llm = _llm()
    agent = get_insurance_agent(llm)
    task = get_insurance_task(agent, payload)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    print("\n[INSURANCE PAYOUT]\n", result)
    return str(result)


def handle_carbon(payload: dict) -> str:
    """
    Calculates carbon credits for food waste prevented.
    Triggers: Carbon MRV Agent → generates VCS-compliant MRV record.
    """
    print("\n[ORCHESTRATOR] Routing CARBON event → Carbon MRV swarm")
    llm = _llm()
    agent = get_carbon_agent(llm)
    task = get_carbon_task(agent, payload)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    print("\n[CARBON MRV RECORD]\n", result)
    return str(result)


def handle_routing(payload: dict) -> str:
    """
    Dispatches a gig driver for distressed asset pickup.
    Triggers: Gig Routing Agent independently.
    """
    print("\n[ORCHESTRATOR] Routing GIG DISPATCH event → Routing Agent swarm")
    llm = _llm()
    agent = get_routing_agent(llm)
    task = get_routing_task(agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    print("\n[ROUTING DISPATCH]\n", result)
    return str(result)


# ── Universal Event Router ─────────────────────────────────────────────────

EVENT_ROUTER = {
    "distress":   handle_distress,
    "pathogen":   handle_pathogen,
    "fraud":      handle_fraud,
    "provenance": handle_provenance,
    "insurance":  handle_insurance,
    "carbon":     handle_carbon,
    "routing":    handle_routing,
}


def handle_event(event: dict) -> str:
    """
    Universal event router.  Dispatches the correct agent swarm based on
    the 'event_type' field in the event dict.

    Parameters
    ----------
    event : dict
        Must contain 'event_type': one of
        'distress', 'pathogen', 'fraud', 'provenance', 'insurance', 'carbon', 'routing'
        Plus event-specific payload fields.

    Returns
    -------
    str  — agent swarm result
    """
    event_type = event.get("event_type", "").lower()
    handler = EVENT_ROUTER.get(event_type)

    if handler is None:
        msg = (
            f"[ORCHESTRATOR] Unknown event_type '{event_type}'. "
            f"Valid types: {list(EVENT_ROUTER.keys())}"
        )
        print(msg)
        return msg

    # For pathogen events, the payload should be a list of detections
    if event_type == "pathogen":
        return handler(event.get("detections", [event]))

    return handler(event)


# ── CLI demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    demo_distress = {
        "event_type":               "distress",
        "asset_id":                 "Pallet-4B-Tomatoes",
        "current_temp_c":           22.5,
        "peak_ethylene_ppm":        3.8,
        "estimated_hours_remaining": 18.0,
        "insured_value_usd":        4200.0,
        "pidr":                     0.0089,
    }

    print("=" * 60)
    print("  KINEXICA ORCHESTRATOR — DEMO")
    print("=" * 60)
    print(json.dumps(demo_distress, indent=2))
    print("\nFiring distress event...\n")
    handle_event(demo_distress)
