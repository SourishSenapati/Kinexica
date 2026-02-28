# pylint: disable=import-error, no-member
"""
Smart Contract deployment logic implementation.
"""
import os
import solcx
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider


def deploy_kinexica_contract(
    asset_id: str, temp: float, ethylene: float, hours: float, price: float,
    grams_methane_saved: int = 0
):
    """
    Simulates or deploys a smart contract directly to a local offline Ethereum node.
    """
    try:
        solcx.install_solc('0.8.19')
    except solcx.exceptions.SolcInstallationError as e:
        print(f"Warning: solc install issue: {e}")

    solcx.set_solc_version('0.8.19')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    contract_path = os.path.join(base_dir, "KinexicaAsset.sol")

    with open(contract_path, "r", encoding="utf-8") as f:
        contract_source = f.read()

    # Compile the smart contract
    compiled_sol = solcx.compile_source(
        contract_source,
        output_values=['abi', 'bin']
    )

    contract_id, contract_interface = compiled_sol.popitem()
    _ = contract_id  # Satisfy linters for this unused val
    abi = contract_interface['abi']
    bytecode = contract_interface['bin']

    # Try connecting to Ganache, fallback to Eth Tester
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
    if not w3.is_connected():
        print(
            "[BLOCKCHAIN] Ganache not detected natively. Booting EthereumTesterProvider...")
        w3 = Web3(EthereumTesterProvider())
    else:
        print("[BLOCKCHAIN] Ganache detected natively!")

    w3.eth.default_account = w3.eth.accounts[0]

    kinexica_asset = w3.eth.contract(abi=abi, bytecode=bytecode)

    # Mint to block
    tx_hash = kinexica_asset.constructor(
        asset_id,
        int(temp * 100),
        int(ethylene * 100),
        int(hours * 100),
        int(price * 100),
        grams_methane_saved
    ).transact()

    # Wait for block
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    return tx_receipt.transactionHash.hex(), tx_receipt.blockNumber


if __name__ == "__main__":
    h, b = deploy_kinexica_contract("TEST", 21.0, 5.0, 100, 2500)
    print(h, b)
