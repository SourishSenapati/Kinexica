"""
Generates synthetic IoT sensor data with thermodynamic decay profiles.
Simulates a baseline degradation with occasional 'spoilage spikes' 
(e.g., a broken AC unit causing temperature and ethylene to rise).
"""

import os
import numpy as np
import pandas as pd


def generate_historical_data(days=30, readings_per_day=24, output_dir=None):
    """
    Generate synthetic IoT sensor data.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'data')
    np.random.seed(42)
    total_readings = days * readings_per_day

    # Time vector (hours)
    time_hours = np.arange(total_readings)

    # Baseline environmental conditions
    base_temp = 15.0  # Celsius
    base_humidity = 85.0  # Percent

    # Chemical kinetics constants (simulated for produce)
    activation_energy = 50000  # J/mol (E_a)
    gas_constant = 8.314       # J/(mol*K) (R)
    pre_exponential = 1e8      # (A)

    # Arrays to hold data
    temps = np.full(total_readings, base_temp) + \
        np.random.normal(0, 0.5, total_readings)
    humidities = np.full(total_readings, base_humidity) + \
        np.random.normal(0, 2, total_readings)
    ethylene_ppm = np.zeros(total_readings)
    shelf_life_remaining = np.zeros(total_readings)

    # Initial shelf life estimation (e.g., 500 hours)
    current_shelf_life = 500.0
    current_ethylene = 0.5  # baseline ppm

    # Inject an anomaly: AC fails at day 15 for 3 days
    ac_failure_start = 15 * readings_per_day
    ac_failure_end = 18 * readings_per_day
    temps[ac_failure_start:ac_failure_end] += np.linspace(
        0, 12, ac_failure_end - ac_failure_start)  # Temp spikes by 12 degrees

    for i in range(total_readings):
        # Convert temp to Kelvin
        temp_k = temps[i] + 273.15

        # Calculate reaction rate constant (k) using Arrhenius
        k = pre_exponential * \
            np.exp(-activation_energy / (gas_constant * temp_k))

        # Ethylene emission correlates with the degradation rate
        current_ethylene += (k * 0.01) + np.random.normal(0, 0.05)
        current_ethylene = max(0.1, current_ethylene)  # Can't go below 0.1
        ethylene_ppm[i] = current_ethylene

        # Shelf life degrades faster when k is higher
        degradation_step = 1.0 + (k * 0.05)
        current_shelf_life -= degradation_step
        shelf_life_remaining[i] = max(0, current_shelf_life)

    # Compile into DataFrame
    df = pd.DataFrame({
        'timestamp_hour': time_hours,
        'temperature_c': np.round(temps, 2),
        'humidity_percent': np.round(humidities, 2),
        'ethylene_ppm': np.round(ethylene_ppm, 3),
        'actual_shelf_life_hours': np.round(shelf_life_remaining, 2)
    })

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'synthetic_sensor_data.csv')
    df.to_csv(file_path, index=False)
    print(
        f"✅ Generated {total_readings} rows of synthetic sensor data at {file_path}")
    print(df.head())


if __name__ == "__main__":
    generate_historical_data()
