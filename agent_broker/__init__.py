"""
agent_broker  — Kinexica AI Agent Swarm
Exports all specialized agents and the master orchestrator.
"""
from agent_broker.negotiation_agent import trigger_negotiation_swarm
from agent_broker.carbon_agent import get_carbon_agent,      get_carbon_task
from agent_broker.biosecurity_agent import get_biosecurity_agent, get_biosecurity_task
from agent_broker.fraud_agent import get_fraud_agent,       get_fraud_task
from agent_broker.provenance_agent import get_provenance_agent,  get_provenance_task
from agent_broker.insurance_agent import get_insurance_agent,   get_insurance_task
from agent_broker.routing_agent import get_routing_agent,     get_routing_task

__all__ = [
    "trigger_negotiation_swarm",
    "get_carbon_agent",      "get_carbon_task",
    "get_biosecurity_agent", "get_biosecurity_task",
    "get_fraud_agent",       "get_fraud_task",
    "get_provenance_agent",  "get_provenance_task",
    "get_insurance_agent",   "get_insurance_task",
    "get_routing_agent",     "get_routing_task",
]
