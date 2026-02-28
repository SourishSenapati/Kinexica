# pylint: disable=import-error, no-member
"""
Module for running the negotiation swarm to mitigate asset degradation costs.
"""
import json
from crewai import Agent, Task, Crew, Process, LLM
from agent_broker.routing_agent import get_routing_agent, get_routing_task


def trigger_negotiation_swarm(payload: dict):
    """
    Takes the JSON payload from the orchestrator and runs the negotiation
    swarm using local LLM.
    """
    print("\n[SWARM WAKE] Anomaly detected. Initiating Negotiation Swarm...\n")

    # Initialize local LLM via CrewAI's designated LLM wrapper
    llm = LLM(model="ollama/llama3")

    # Phase 2: Agent 1 - The Kinetics Assessor
    assessor = Agent(
        role='Kinetics Assessor',
        goal='Analyze biological degradation data and format emergency liquidation reports',
        backstory=(
            "You are a senior Chemical Process Engineer and Logistics Assessor. "
            "You do not converse; you only analyze biological degradation data based "
            "on thermodynamics and format emergency reports for distressed assets."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Phase 3: Agent 2 - The Autonomous Broker
    broker = Agent(
        role='Autonomous Broker',
        goal='Recover sunk costs from rapidly degrading agricultural assets by '
             'negotiating smart contracts with secondary buyers.',
        backstory=(
            "You are a ruthless, high-speed commodities broker. Your sole objective "
            "is to maximize value recovery from rapidly degrading agricultural "
            "assets by negotiating smart contracts with secondary buyers before "
            "the asset value hits zero."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Assessor Task
    assess_task = Task(
        description=(
            f"Analyze the following distressed asset metrics:\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Calculate the severity of the spoilage kinetics. If ethylene levels "
            "indicate an irreversible ripening cascade, immediately classify "
            "the asset as 'Tier 1 Distressed'.\n\n"
            "Generate an 'Asset Liquidation Profile' containing exactly:\n"
            "- Asset Name (e.g., from the payload)\n"
            "- Current viable hours remaining.\n"
            "- Recommended secondary use-case (e.g., Target: Bio-feed processing, "
            "Aquaculture feed, or immediate local discount retail)."
        ),
        expected_output="An Asset Liquidation Profile outlining distress tier, "
                        "viable hours remaining, and recommended secondary use-case.",
        agent=assessor
    )

    # Broker Task
    broker_task = Task(
        description=(
            "Accept the Liquidation Profile from the Assessor. Cross-reference "
            "the recommended secondary use-case with potential local buyers "
            "(e.g., local aquaculture facility needing organic input).\n\n"
            "Apply a time-decay pricing model: e.g., '72 hours remaining = 40% "
            "discount, 24 hours remaining = 80% discount'.\n\n"
            "Draft a targeted, urgent communication/smart contract summary to "
            "the matched buyer. The message must state the exact biological state "
            "of the asset, the dynamically calculated price, and an ultimatum "
            "for immediate pickup."
        ),
        expected_output="A finalized smart contract/sale agreement drafted for "
                        "the secondary buyer, outputted to the terminal.",
        agent=broker
    )

    # Phase 4: Routing Agent
    routing_agent = get_routing_agent(llm)
    r_task = get_routing_task(routing_agent)

    # Instantiate the Crew
    negotiation_crew = Crew(
        agents=[assessor, broker, routing_agent],
        tasks=[assess_task, broker_task, r_task],
        process=Process.sequential
    )

    # Kickoff the swarm
    result = negotiation_crew.kickoff()

    print("\n================================================")
    print("FINAL SMART CONTRACT / LIQUIDATION AGREEMENT:")
    print("================================================")
    print(result)
    return result
