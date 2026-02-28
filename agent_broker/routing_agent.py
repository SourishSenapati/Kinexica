# pylint: disable=import-error, no-member
"""
Routing Agent module for processing data.
"""

from crewai import Agent, Task
from agent_broker.tools import (
    save_contract_to_ledger, broadcast_webhook, mint_smart_contract,
    calculate_biomass_carbon_yield, dispatch_last_mile_gig_driver,
    calculate_dynamic_shelf_price
)


def get_routing_agent(llm):
    """
    Initializes the Routing Agent which finalizes the transaction.
    """
    return Agent(
        role='Supply Chain Routing & Biotech Aggregation Agent',
        goal='Maximize profit via Advanced Bio-Manufacturing routes, distribute '
             'gig-economy rescue missions, and adjust dynamic retail shelf prices.',
        backstory=(
            "You are the ultimate logistics & monetization core in the supply chain. "
            "You don't just dump rotting food—you find exact carbon substrates for "
            "precision fermentation labs. If an asset is dying, you immediately dispatch "
            "a local Uber Freight gig-driver via the Kinexica Rescue API. Meanwhile, "
            "you update the Electronic Shelf Labels in grocery stores to deeply discount "
            "produce with less than 18 hours of shelf life to recover pure retail margin."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[
            save_contract_to_ledger, broadcast_webhook, mint_smart_contract,
            calculate_biomass_carbon_yield, dispatch_last_mile_gig_driver,
            calculate_dynamic_shelf_price
        ],
        llm=llm
    )


def get_routing_task(routing_agent):
    """
    Returns the task for the routing agent to execute.
    """
    return Task(
        description=(
            "Accept the completed Smart Contract details. You MUST execute the following new monetization steps:\n"
            "1. Use 'Calculate Dynamic Shelf Price' (if hours < 18) to push an automated retail markdown.\n"
            "2. If completely degraded, use 'Calculate Biomass Carbon Yield' to market the asset to Precision Fermentation Lab XYZ.\n"
            "3. Use 'Mint Immutable Smart Contract' to permanently secure the biological asset degradation metrics.\n"
            "4. Use 'Dispatch Gig Driver (Kinexica Rescue API)' to trigger last-mile movement via Uber Freight.\n"
            "5. Save all transaction receipts to the Ledger.\n"
        ),
        expected_output=(
            "A comprehensive execution trace confirming shelf prices dynamically updated, "
            "freight gig-drivers dispatched, and high-margin Biotech routing completed, all secured on Web3."
        ),
        agent=routing_agent
    )
