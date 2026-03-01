"""
Kinexica — Chemical Fraud Investigator Agent
Processes Visual-PINN fraud detections, builds an evidence package,
escalates to enforcement authorities, and drafts consumer alerts.
"""
# pylint: disable=import-error
from crewai import Agent, Task, LLM


def get_fraud_agent(llm: LLM) -> Agent:
    return Agent(
        role="Chemical Fraud Investigator",
        goal=(
            "Build court-admissible evidence packages for detected chemical food fraud cases, "
            "recommend confirmatory laboratory tests, and draft regulatory enforcement notices "
            "compliant with FSSAI Act 2006, Prevention of Food Adulteration Act, and Codex CAC/GL 33."
        ),
        backstory=(
            "You are a senior food forensics scientist and legal compliance officer. "
            "You have processed over 2,000 food adulteration cases for FSSAI and NAAC labs. "
            "You specialize in calcium carbide (CaC₂) ripening fraud, formalin preservation, "
            "synthetic dye injection, and heavy metal contamination. You know which confirmatory "
            "tests are legally admissible in Indian courts (AgNO₃ test, HPLC, ICP-OES, AOAC 967.19) "
            "and how to draft evidence packages that hold up to judicial scrutiny."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def get_fraud_task(agent: Agent, fraud_result: dict) -> Task:
    fraud_types = fraud_result.get("fraud_types", [])
    confidence = fraud_result.get("fraud_confidence", 0.0)
    asset_id = fraud_result.get("asset_id",    "unknown")
    commodity = fraud_result.get("commodity",   "produce")
    vendor = fraud_result.get("vendor",      "unknown")
    location = fraud_result.get("location",    "unknown")
    severity = fraud_result.get("severity",    "Medium")

    test_map = {
        "calcium carbide": "AgNO₃ precipitation test (field) + AAS/ICP-OES (lab) + GC-MS for acetylene",
        "formalin":        "AOAC 967.19 aldehyde titration + LC-MS/MS",
        "dye injection":   "HPLC + TLC for Sudan Red I-IV, Metanil Yellow, Rhodamine B",
        "wax coating":     "Hexane extraction + GC-FID for petroleum hydrocarbons",
        "heavy metal":     "ICP-OES / ICP-MS for Pb, Cd, Hg, As (CODEX CXS 193-1995)",
        "ethephon":        "GC-NPD or LC-MS/MS for 2-chloroethylphosphonic acid",
    }

    recommended_tests = []
    for f_type in fraud_types:
        for key, test in test_map.items():
            if key in f_type.lower() and test not in recommended_tests:
                recommended_tests.append(test)
    if not recommended_tests:
        recommended_tests = [
            "Comprehensive food adulteration panel (FSSAI Schedule IV)"]

    return Task(
        description=(
            f"FRAUD DETECTION REPORT\n"
            f"Asset ID  : {asset_id}\n"
            f"Commodity : {commodity}\n"
            f"Vendor    : {vendor}\n"
            f"Location  : {location}\n"
            f"Severity  : {severity}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Fraud types: {', '.join(fraud_types)}\n\n"
            f"Recommended confirmatory tests:\n"
            + "\n".join(f"  • {t}" for t in recommended_tests)
            + "\n\nTasks:\n"
            "1. Assess evidentiary strength of the visual detection (confidence level).\n"
            "2. Draft a legal EVIDENCE PACKAGE containing:\n"
            "   - Case summary with asset metadata\n"
            "   - Fraud type taxonomy and applicable IPC sections (272, 273, 420 IPC)\n"
            "   - Confirmatory tests with NABL-accredited labs list\n"
            "   - Chain of custody protocol for sample preservation\n"
            "3. Draft a CONSUMER ALERT followng FSSAI public notice template.\n"
            "4. Draft REGULATORY NOTICE to:\n"
            "   - FSSAI Food Safety Commissioner\n"
            "   - State Food Safety Designated Officer (FSDO)\n"
            "   - Consumer Affairs Ministry (if dye/carbide — mass market impact)\n"
            "5. Recommend immediate enforcement action: Seizure / FIR / License suspension."
        ),
        expected_output=(
            "Full evidence package with case summary, IPC citations, confirmatory tests, "
            "consumer alert draft, and regulatory enforcement notice."
        ),
        agent=agent,
    )
