"""
Routing Agent module for processing data.
"""

from crewai import Agent, Task
from agent_broker.tools import save_contract_to_ledger, broadcast_webhook, mint_smart_contract


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
            "is brokered, you ensure that the transaction guarantees are saved, "
            "physical movers are notified, and an immutable proof-of-kinetics "
            "smart contract is successfully minted on the local blockchain."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[save_contract_to_ledger, broadcast_webhook, mint_smart_contract],
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
            "Then, you MUST use the 'Mint Immutable Smart Contract' tool to permanently secure "
            "the asset degradation metrics and sale price into the local Ganache block. "
            "Lastly, you MUST use the 'Broadcast Webhook to Supply Chain' tool to notify "
            "drivers of the pickup."
        ),
        expected_output="Confirmation that the smart contract was minted on Web3, the ledger has been updated, and webhooks fired.",
        agent=routing_agent
    )
