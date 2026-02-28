# pylint: disable=import-error, no-member, unexpected-keyword-arg, broad-except, too-many-function-args, wrong-import-position, wrong-import-order, line-too-long, missing-function-docstring
"""
SpoilSense Edge Client: Real-time UI dashboard rendered using Flet.
Includes Phase 12 Monetization Engine 3-Tiers: B2B QA Gateway, B2C Mobile Lens, B2G Heatmap.
"""
from pinn_engine.visual_pinn import analyze_lesion_kinetics
from pinn_engine.syndi_trust import apply_synthid_watermark
import asyncio
import os
import requests
import flet as ft
from sys import path
path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


API_BASE_URL = os.getenv("API_BASE_URL", "https://real-cougars-attack.loca.lt")

# Hierarchy of Darkness palette
BG_DARK_0 = "#040405"  # Deepest background
BG_DARK_1 = "#09090b"  # App background
BG_DARK_2 = "#18181b"  # Card background
BG_DARK_3 = "#27272a"  # Elevated hover / border
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#10b981"
ACCENT_PURPLE = "#8b5cf6"
ACCENT_RED = "#ef4444"
TEXT_PRIMARY = "#f4f4f5"
TEXT_SECONDARY = "#a1a1aa"


def main(page: ft.Page):
    # Force mobile dimensions and responsive scrolling utilizing Hierarchy of Darkness
    page.title = "Kinexica Edge Client"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = BG_DARK_1
    page.window_width = 400        # Constrain width for desktop testing
    page.window_height = 800       # Constrain height for desktop testing
    page.scroll = ft.ScrollMode.ADAPTIVE  # Crucial for mobile scrolling
    page.padding = 20
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    page.fonts = {
        "Inter": "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap"}
    page.theme = ft.Theme(font_family="Inter", color_scheme_seed=ACCENT_BLUE)

    # === Tier 1: Kinexica B2B QA Gateway ===
    title = ft.Text("SpoilSense Telemetry", size=26,
                    weight=ft.FontWeight.W_800, color=TEXT_PRIMARY)
    subtitle = ft.Text("B2B Evaluation Hub", size=14, color=TEXT_SECONDARY)

    asset_id_text = ft.Text("Asset ID: ---", size=14, color=TEXT_SECONDARY)
    temp_text = ft.Text("Temp: -- °C", size=15, color=TEXT_PRIMARY)
    ethylene_text = ft.Text("Ethylene: -- ppm", size=15, color=TEXT_PRIMARY)
    shelf_life_text = ft.Text(
        "Est. Shelf Life: -- h", size=16, weight=ft.FontWeight.BOLD, color=ACCENT_GREEN)

    status_text = ft.Text("STATUS: AWAITING...", size=16,
                          weight=ft.FontWeight.W_800, color=TEXT_SECONDARY)
    tx_hash_text = ft.Text(
        "", size=12, weight=ft.FontWeight.BOLD, color=ACCENT_PURPLE, visible=False)

    data_col = ft.Column(
        [temp_text, ethylene_text, shelf_life_text], spacing=8)

    status_card_content = ft.Column(
        [asset_id_text, ft.Divider(color=BG_DARK_3), data_col, ft.Divider(
            height=20, color="transparent"), status_text, tx_hash_text],
        alignment=ft.MainAxisAlignment.START, spacing=8
    )

    status_card = ft.Container(
        content=status_card_content,
        width=340,
        bgcolor=BG_DARK_2,
        border=ft.border.all(1, BG_DARK_3),
        border_radius=16,
        padding=25,
        alignment=ft.Alignment(0, 0),
        shadow=ft.BoxShadow(spread_radius=1, blur_radius=20,
                            color=BG_DARK_0, offset=ft.Offset(0, 8))
    )
    status_card.animate = ft.Animation(500, ft.AnimationCurve.EASE_OUT)

    b2b_start_btn = ft.Container(
        content=ft.Text("INITIALIZE TELEMETRY", color=TEXT_PRIMARY,
                        weight=ft.FontWeight.BOLD, size=14),
        bgcolor=ACCENT_BLUE, border_radius=12, padding=ft.padding.symmetric(15, 30),
        ink=True, on_click=lambda e: requests.post(f"{API_BASE_URL}/start-monitoring", timeout=2),
        shadow=ft.BoxShadow(
            blur_radius=10, color="#1a3b82f6", offset=ft.Offset(0, 4))
    )

    # === Tier 2: Kinexica B2C Mobile Lens (Zero-Hardware Fallback) ===
    lens_title = ft.Text("Edge Optical Lens", size=26,
                         weight=ft.FontWeight.W_800, color=TEXT_PRIMARY)
    lens_sub = ft.Text("B2C Deep Inspection", size=14, color=TEXT_SECONDARY)

    img_path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "..", "data", "synth_images", "tomato_pathogenic_1772284284661.png"))
    out_path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "..", "data", "synth_images", "tomato_watermarked.png"))

    lens_img = ft.Image(src=img_path, width=320, height=320, fit=ft.BoxFit.COVER,
                        visible=False, border_radius=ft.border_radius.all(16))
    lens_image_container = ft.Container(
        content=lens_img, bgcolor=BG_DARK_2, width=320, height=320, border_radius=16, border=ft.border.all(1, BG_DARK_3),
        shadow=ft.BoxShadow(blur_radius=20, color=BG_DARK_0,
                            offset=ft.Offset(0, 8))
    )

    lens_status = ft.Text("Module Initialized. Ready for matrix capture.",
                          size=14, color=TEXT_SECONDARY, text_align=ft.TextAlign.CENTER)

    lens_gps = ft.Text("LOC: 40.71° N, -74.00° W (Truncated)\nENV: Open-Meteo | 22°C",
                       size=11, color=BG_DARK_3, text_align=ft.TextAlign.CENTER)

    def trigger_lens_scan(_):
        lens_img.visible = True
        lens_status.color = ACCENT_BLUE
        lens_status.value = "Executing Diffusion Kinetics..."
        page.update()

        try:
            result = analyze_lesion_kinetics(img_path, crop_archetype=1)
            apply_synthid_watermark(
                img_path, out_path, "SCAN-KXT", result.get('classification', 'Unknown'))

            if result.get("color") == "red":
                lens_status.color = ACCENT_RED
                lens_status.value = "CRITICAL: Pathogenic Matrix Match (Botrytis)\n[SyndiTrust Secured]"
            elif result.get("color") == "purple":
                lens_status.color = ACCENT_PURPLE
                lens_status.value = "ALERT: Chemical Adulteration (Calcium Carbide)\n[SyndiTrust Secured]"
            else:
                lens_status.color = ACCENT_GREEN
                lens_status.value = "VERIFIED: Baseline Biological Kinetics\n[SyndiTrust Secured]"
        except Exception as e:
            pass
        page.update()

    scan_btn = ft.Container(
        content=ft.Row([ft.Icon(ft.icons.CAMERA_OUTLINED, color=TEXT_PRIMARY), ft.Text(
            "CAPTURE MATRIX", color=TEXT_PRIMARY, weight=ft.FontWeight.BOLD, size=14)], alignment=ft.MainAxisAlignment.CENTER),
        width=320, bgcolor=ACCENT_PURPLE, border_radius=12, padding=15, ink=True, on_click=trigger_lens_scan,
        shadow=ft.BoxShadow(
            blur_radius=15, color="#228b5cf6", offset=ft.Offset(0, 4))
    )

    # === Tier 3: B2G Bio-Security Heatmap ===
    b2g_title = ft.Text("Bio-Security Maps", size=26,
                        weight=ft.FontWeight.W_800, color=TEXT_PRIMARY)
    b2g_sub = ft.Text("B2G Intelligence Hub", size=14, color=TEXT_SECONDARY)
    b2g_data = ft.Text("Monitoring secure geographic nodes...",
                       size=14, color=TEXT_SECONDARY, italic=True)

    b2g_alerts = ft.ListView(expand=True, spacing=15, height=300)

    def construct_alert(icon, text, badge_color):
        return ft.Container(
            content=ft.Row([ft.Icon(icon, color=badge_color, size=20), ft.Text(
                text, size=13, color=TEXT_PRIMARY, expand=True)]),
            bgcolor=BG_DARK_2, border=ft.border.all(1, BG_DARK_3), border_radius=12, padding=15
        )

    b2g_alerts.controls.append(construct_alert(
        ft.icons.PUPCATCH_OUTLINED, "Botrytis cinerea outbreak detected in Sector 4", ACCENT_RED))
    b2g_alerts.controls.append(construct_alert(
        ft.icons.NATURE_OUTLINED, "Elevated Penicillium traces in local watershed", "#fbbf24"))

    # TABS
    t = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        unselected_label_color=TEXT_SECONDARY,
        label_color=ACCENT_BLUE,
        indicator_color=ACCENT_BLUE,
        divider_color=BG_DARK_3,
        overlay_color=BG_DARK_2,
        tabs=[
            ft.Tab(
                text="HUB",
                content=ft.Container(
                    ft.Column([ft.Divider(height=10, color="transparent"), title, subtitle, ft.Divider(height=20, color="transparent"), status_card, ft.Divider(
                        height=10, color="transparent"), b2b_start_btn], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=10
                )
            ),
            ft.Tab(
                text="LENS",
                content=ft.Container(
                    ft.Column([ft.Divider(height=10, color="transparent"), lens_title, lens_sub, ft.Divider(height=10, color="transparent"), lens_image_container, ft.Divider(
                        height=10, color="transparent"), scan_btn, lens_status, ft.Divider(height=10, color="transparent"), lens_gps], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=10
                )
            ),
            ft.Tab(
                text="MAPS",
                content=ft.Container(
                    ft.Column([ft.Divider(height=10, color="transparent"), b2g_title, b2g_sub, ft.Divider(height=20, color="transparent"),
                              b2g_data, b2g_alerts], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=10
                )
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
                        asset_id_text.value = f"ID: {data['asset_id']}"
                        temp_text.value = f"{data['current_temp_c']} °C"
                        ethylene_text.value = f"{data['ethylene_ppm']} ppm"
                        shelf_life_text.value = f"{data['estimated_shelf_life_h']:.1f} hrs"
                        status = data['status']
                        status_text.value = f"STATUS: {status.upper()}"

                        if status == "Stable":
                            status_card.border = ft.border.all(1, ACCENT_GREEN)
                            status_text.color = ACCENT_GREEN
                            is_flashing = False
                            tx_hash_text.visible = False
                        elif status == "Liquidated" and data.get("tx_hash"):
                            status_card.border = ft.border.all(
                                2, ACCENT_PURPLE)
                            status_text.color = ACCENT_PURPLE
                            is_flashing = False
                            tx_hash_text.value = f"TxHash: {data['tx_hash'][:12]}... | Web3 Settled"
                            tx_hash_text.visible = True
                            status_text.value = "LEDGER FINALIZED"
                        elif status in ("Distressed", "Liquidated"):
                            is_flashing = not is_flashing
                            status_card.border = ft.border.all(
                                2, ACCENT_RED if is_flashing else BG_DARK_3)
                            status_text.color = ACCENT_RED
                    else:
                        status_text.value = "STATUS: UNAVAILABLE"
                else:
                    status_text.value = "STATUS: OFFLINE"
            except Exception:
                pass
            page.update()
            await asyncio.sleep(1.0)

    page.run_task(poll_backend)


if __name__ == "__main__":
    ft.app(target=main)
