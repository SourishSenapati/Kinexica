// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract KinexicaAsset {
    string public assetId;
    int256 public terminalTemperature; // Multiplied by 100
    int256 public peakEthylenePpm; // Multiplied by 100
    int256 public remainingHours; // Multiplied by 100
    int256 public liquidatedPrice; // Multiplied by 100

    constructor(
        string memory _assetId,
        int256 _temp,
        int256 _ethylene,
        int256 _hours,
        int256 _price
    ) {
        assetId = _assetId;
        terminalTemperature = _temp;
        peakEthylenePpm = _ethylene;
        remainingHours = _hours;
        liquidatedPrice = _price;
    }
}
