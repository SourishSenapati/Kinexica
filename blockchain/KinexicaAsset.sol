// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract KinexicaAsset {
    string public assetId;
    int256 public terminalTemperature; // Multiplied by 100
    int256 public peakEthylenePpm; // Multiplied by 100
    int256 public remainingHours; // Multiplied by 100
    int256 public liquidatedPrice; // Multiplied by 100
    
    // Phase 2: MRV Protocol - Carbon Credit Engine
    uint256 public carbonOffsetMethaneGrams; // Methane offset tracked
    mapping(address => uint256) public kctBalances; // Kinexica Carbon Token (KCT) ERC-20 Compliant Balance map
    event Transfer(address indexed from, address indexed to, uint256 value); // ERC-20 standard event
    event CarbonTokenMinted(address indexed upcycler, uint256 gramsMethaneSaved);

    constructor(
        string memory _assetId,
        int256 _temp,
        int256 _ethylene,
        int256 _hours,
        int256 _price,
        uint256 _gramsMethaneSaved
    ) {
        assetId = _assetId;
        terminalTemperature = _temp;
        peakEthylenePpm = _ethylene;
        remainingHours = _hours;
        liquidatedPrice = _price;
        carbonOffsetMethaneGrams = _gramsMethaneSaved;

        // Mint ERC-20 KCT tokens equivalent to grams of methane prevented directly to the deployer
        kctBalances[msg.sender] += _gramsMethaneSaved;
        emit Transfer(address(0), msg.sender, _gramsMethaneSaved);
        emit CarbonTokenMinted(msg.sender, _gramsMethaneSaved);
    }
}
