"""
Kinexica — Bio-Security & Pathogen Threat Intelligence Agent
Maps pathogen outbreaks across supply chain nodes at district granularity
and issues regulatory alerts for government food safety dashboards (B2G).
"""
# pylint: disable=import-error
from crewai import Agent, Task, LLM


def get_biosecurity_agent(llm: LLM) -> Agent:
    return Agent(
        role="Bio-Security Intelligence Analyst",
        goal=(
            "Map foodborne pathogen outbreak vectors across supply chain nodes, "
            "identify cross-contamination corridors, and issue FSSAI/FDA-compliant "
            "bio-security alerts for government food safety authorities."
        ),
        backstory=(
            "You are a senior epidemiologist and food safety scientist trained at WHO FOS "
            "and ICMR. You specialize in molecular outbreak analysis (PFGE, WGS) and "
            "supply chain network disease propagation modelling. You work with Kinexica's "
            "Visual-PINN engine output to map where Botrytis, Aspergillus, Penicillium, "
            "and bacterial soft rot detections cluster geographically, then escalate to "
            "government regulators before an outbreak becomes a public health emergency."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def get_biosecurity_task(agent: Agent, detections: list) -> Task:
    """
    detections: list of dicts from Visual-PINN, each containing
    {asset_id, location_district, pathogen_species, severity, timestamp}
    """
    n_critical = sum(1 for d in detections if d.get("severity") == "Critical")
    n_high = sum(1 for d in detections if d.get("severity") == "High")
    species = list(
        {s for d in detections for s in d.get("pathogen_species", [])})

    return Task(
        description=(
            f"Active detection batch: {len(detections)} events\n"
            f"Critical severity: {n_critical}  |  High severity: {n_high}\n"
            f"Unique species detected: {species}\n\n"
            "Detection records:\n"
            + "\n".join(
                f"  [{d.get('asset_id', '?')}] {d.get('location_district', '?')} — "
                f"{', '.join(d.get('pathogen_species', ['Unknown']))} "
                f"(Severity: {d.get('severity', '?')})"
                for d in detections[:20]   # cap to 20 for prompt size
            )
            + "\n\nTasks:\n"
            "1. Identify spatial clustering — list districts with ≥2 concurrent Critical detections.\n"
            "2. Assess cross-contamination probability between adjacent nodes.\n"
            "3. If Aspergillus detected → mandatory aflatoxin outbreak protocol (CODEX CXG 31).\n"
            "4. Draft a Bio-Security Alert in FSSAI FSMS format:\n"
            "   - Threat level: GREEN / AMBER / RED\n"
            "   - Affected districts\n"
            "   - Species taxonomy\n"
            "   - Recommended quarantine radius\n"
            "   - Mandatory lab confirmation tests\n"
            "5. List regulatory bodies to notify: FSSAI, FDA, APEDA, State FSSA."
        ),
        expected_output=(
            "A structured FSSAI bio-security alert with threat level, affected zones, "
            "pathogen taxonomy, quarantine recommendations, and regulatory notification list."
        ),
        agent=agent,
    )
