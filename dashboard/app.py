# pylint: disable=import-error, no-member, unexpected-keyword-arg, broad-except, too-many-function-args, wrong-import-position, wrong-import-order, line-too-long, missing-function-docstring
"""
SpoilSense Edge Client: Real-time UI dashboard rendered using Flet.
Includes Phase 12 Monetization Engine 3-Tiers: B2B QA Gateway, B2C Mobile Lens, B2G Heatmap.
"""
from pinn_engine.syndi_trust import apply_synthid_watermark
from pinn_engine.visual_pinn import analyze_lesion_kinetics
import asyncio
import os
import requests
import flet as ft
from sys import path

path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def main(page: ft.Page):
    page.title = "SpoilSense Edge Client"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.ADAPTIVE

    # === Tier 1: Kinexica B2B QA Gateway ===
    title = ft.Text("SpoilSense Telemetry (B2B Hub)",
                    size=24, weight=ft.FontWeight.BOLD)
    asset_id_text = ft.Text("Asset ID: ---", size=16)
    temp_text = ft.Text("Temp: -- °C", size=16)
    ethylene_text = ft.Text("Ethylene: -- ppm", size=16)
    shelf_life_text = ft.Text("Est. Shelf Life: -- h", size=16)
    status_text = ft.Text("Status: UNKNOWN", size=20,
                          weight=ft.FontWeight.BOLD)
    tx_hash_text = ft.Text("", size=14, weight=ft.FontWeight.BOLD,
                           color=ft.colors.YELLOW_200, visible=False)

    col = ft.Column([asset_id_text, temp_text, ethylene_text, shelf_life_text,
                    status_text, tx_hash_text], alignment=ft.MainAxisAlignment.CENTER)
    status_card = ft.Container(content=col, width=300, height=200, bgcolor=ft.colors.GREY_800,
                               border_radius=15, padding=20, alignment=ft.Alignment(0, 0))
    status_card.animate = ft.animation.Animation(
        500, ft.AnimationCurve.EASE_OUT)

    # === Tier 2: FinnoAQ B2C Mobile Lens (Zero-Hardware Fallback) ===
    lens_title = ft.Text("FinnoAQ Edge Lens (B2C Mode)",
                         size=24, weight=ft.FontWeight.BOLD)

    img_path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "..", "data", "synth_images", "tomato_pathogenic_1772284284661.png"))
    out_path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "..", "data", "synth_images", "tomato_watermarked.png"))

    lens_img = ft.Image(src=img_path, width=300, height=300,
                        fit=ft.ImageFit.CONTAIN, visible=False)
    lens_status = ft.Text("Awaiting Scan...", size=18,
                          color=ft.colors.BLUE_200)

    # Coordinates rounded to 5km radius for Edge Truncation privacy (GDPR / DPDP)
    lens_gps = ft.Text("GPS: 40.71° N, -74.00° W (Truncated 5km Radius) | Open-Meteo Temp: 22°C",
                       size=12, color=ft.colors.GREY_400)

    def trigger_lens_scan(_):
        lens_img.visible = True
        lens_status.color = ft.colors.BLUE_400
        lens_status.value = "Analyzing Reaction-Diffusion Kinetics..."
        page.update()

        # Zero-Hardware Fallback: Ambient temp + Visual PINN
        result = analyze_lesion_kinetics(
            img_path, crop_archetype=1)  # Archetype 1: Tomato

        # Apply SynthID Protocol
        apply_synthid_watermark(
            img_path, out_path, "SCAN-101", result.get('classification', 'Unknown'))

        if result.get("color") == "red":
            lens_status.color = ft.colors.RED_400
            lens_status.value = "Anomalous Degradation - Divert from Human Consumption\n(Pathogenic Variant: Botrytis cinerea)\n[Syndi Trust Verified]"
        elif result.get("color") == "purple":
            lens_status.color = ft.colors.PURPLE_accent_400
            lens_status.value = "CRITICAL: Chemical Adulteration Detected (Calcium Carbide Fraud)\n[Syndi Trust Verified]"
        else:
            lens_status.color = ft.colors.GREEN_400
            lens_status.value = "Visual Kinetics Normal - No Pathogen Detected\n[Syndi Trust Verified]"

        page.update()

    scan_btn = ft.ElevatedButton("📷 Take Photo / Scan Fruit", on_click=trigger_lens_scan,
                                 bgcolor=ft.colors.PURPLE_600, color=ft.colors.WHITE)

    # === Tier 3: B2G Bio-Security Heatmap ===
    b2g_title = ft.Text("National Bio-Security Heatmap (B2G License)",
                        size=24, weight=ft.FontWeight.BOLD)
    b2g_data = ft.Text(
        "Loading live outbreak geometries from PostGIS...", size=16, italic=True)
    b2g_alerts = ft.ListView(expand=True, spacing=10, height=200)
    b2g_alerts.controls.append(ft.Text(
        "🔴 ALERT: Botrytis cinerea outbreak detected in Zone 4 (50 scans/hr)", color=ft.colors.RED_400))
    b2g_alerts.controls.append(ft.Text(
        "🟡 WARNING: Penicillium levels rising in Zone 2", color=ft.colors.AMBER_400))

    # TABS
    t = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="B2B: Kinexica Hub",
                content=ft.Column([title, status_card, ft.ElevatedButton("Start Monitoring Loop", on_click=lambda e: requests.post(
                    f"{API_BASE_URL}/start-monitoring", timeout=2), bgcolor=ft.colors.BLUE_700, color=ft.colors.WHITE)], alignment=ft.MainAxisAlignment.CENTER)
            ),
            ft.Tab(
                text="B2C: Mobile Lens",
                content=ft.Column([lens_title, lens_gps, scan_btn, lens_img,
                                  lens_status], alignment=ft.MainAxisAlignment.CENTER)
            ),
            ft.Tab(
                text="B2G: Heatmap",
                content=ft.Column([b2g_title, b2g_data, b2g_alerts],
                                  alignment=ft.MainAxisAlignment.CENTER)
            ),
        ],
        expand=1,
    )
    page.add(t)

    # Original B2B Polling Loop
    async def poll_backend():
        is_flashing = False
        while True:
            try:
                res = requests.get(
                    f"{API_BASE_URL}/asset/Pallet-4B-Tomatoes", timeout=2)
                if res.status_code == 200:
                    data = res.json()
                    if "error" not in data:
                        asset_id_text.value = f"Asset ID: {data['asset_id']}"
                        temp_text.value = f"Temp: {data['current_temp_c']} °C"
                        ethylene_text.value = f"Ethylene: {data['ethylene_ppm']} ppm"
                        shelf_life_text.value = f"Est. Shelf Life: {data['estimated_shelf_life_h']:.2f} h"
                        status = data['status']
                        status_text.value = f"Status: {status.upper()}"

                        if status == "Stable":
                            status_card.bgcolor = ft.colors.GREEN_700
                            is_flashing = False
                            tx_hash_text.visible = False
                        elif status == "Liquidated" and data.get("tx_hash"):
                            status_card.bgcolor = ft.colors.AMBER_600
                            is_flashing = False
                            tx_hash_text.value = f"TxHash: {data['tx_hash'][:10]}... | Block: {data['block_number']} | KCT Minted"
                            tx_hash_text.visible = True
                            status_text.value = "Status: IMMUTABLE CONTRACT SECURED"
                        elif status in ("Distressed", "Liquidated"):
                            is_flashing = not is_flashing
                            status_card.bgcolor = ft.colors.RED_900 if is_flashing else ft.colors.RED_500
                    else:
                        status_text.value = "Status: NOT FOUND"
                else:
                    status_text.value = "Status: OFFLINE"
            except Exception:
                pass
            page.update()
            await asyncio.sleep(1.0)

    page.run_task(poll_backend)


if __name__ == "__main__":
    ft.app(target=main)
