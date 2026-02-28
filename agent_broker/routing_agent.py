"""
Routing Agent module for processing data.
"""

from crewai import Agent, Task
from agent_broker.tools import save_contract_to_ledger, broadcast_webhook


def get_routing_agent(llm):
    """
    Initializes the Routing Agent which finalizes the transaction.
    """
    return Agent(
        role='Supply Chain Routing Agent',
        goal='Take the completed smart contract from the Broker and distribute '
             'it securely to the ledger and notify logistics.',
        backstory=(
            "You are the final node in the intelligent supply chain. Once a deal "
            "is brokered, you ensure that the transaction guarantees are saved "
            "and all physical movers are notified instantly."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[save_contract_to_ledger, broadcast_webhook],
        llm=llm
    )


def get_routing_task(routing_agent):
    """
    Returns the task for the routing agent to execute.
    """
    return Task(
        description=(
            "Accept the completed Smart Contract / Liquidation Agreement. "
            "You MUST use the 'Save Smart Contract to Ledger' tool to save the exact text. "
            "Then, you MUST use the 'Broadcast Webhook to Supply Chain' tool to notify "
            "drivers of the pickup."
        ),
        expected_output="Confirmation that the ledger has been updated and webhooks fired.",
        agent=routing_agent
    )
