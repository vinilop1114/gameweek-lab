"""Briefing compacto en Markdown para redactar posts.

Por qué existe: el Project de claude.ai leía `players_scored.csv`
(592 jugadores × 32 columnas, ~39k tokens) para armar un post que usa
~20 jugadores. Además exponía dos trampas — jugadores con pocos minutos
y tasas por 90' infladas, y columnas crudas que hay que saber
interpretar.

Este archivo resuelve las tres cosas: ~10x menos tokens, ya filtrado por
muestra confiable, y en Markdown (más fácil de leer para un modelo que un
CSV ancho). Se publica en GitHub como el resto de los datos.
"""

from datetime import datetime, timezone

import pandas as pd

from gameweek_lab.analysis import (
    HORIZON_GAMEWEEKS,
    MIN_MINUTES_FOR_RANKING,
    captaincy_picks,
    top_differentials,
)
from gameweek_lab.calibration import _load_history
from gameweek_lab.transfer_advisor import _load_base_state, preview_base_transfers
from gameweek_lab.config import DATA_PROCESSED_DIR

BRIEFING_PATH = DATA_PROCESSED_DIR / "briefing.md"


def _format_squad_table(starters: pd.DataFrame, bench: pd.DataFrame) -> str:
    lines = [
        "| Rol | Jugador | Equipo | Pos | Precio | xP | Techo | P(haul) | xP 4GW | Próximo rival | Balón parado |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for team, is_starter in ((starters, True), (bench, False)):
        for p in team.itertuples():
            if is_starter:
                role = "**Capitán**" if p.is_captain else ("Vice" if p.is_vice_captain else "Titular")
            else:
                role = f"Banco {int(p.bench_order)}"
            venue = "casa" if p.next_is_home else "fuera"
            set_pieces = p.set_piece_duties if p.set_piece_duties else "—"
            lines.append(
                f"| {role} | {p.web_name} | {p.team_name} | {p.position} | £{p.now_cost}m | "
                f"{p.xp_next} | {p.xp_ceiling} | {p.haul_probability:.0%} | {p.xp_horizon} | "
                f"{p.next_opponent} ({venue}) | {set_pieces} |"
            )
    return "\n".join(lines)


def _squad_from_export(squads: pd.DataFrame, squad_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    team = squads[squads["squad_type"] == squad_type]
    starters = team[team["role"] == "Titular"].sort_values("xp_next", ascending=False)
    bench = team[team["role"] == "Banco"].sort_values("bench_order")
    return starters, bench


def _transfer_section(squads: pd.DataFrame, proposed_log: list[str]) -> list[str]:
    """Las transferencias: la propuesta de hoy y, si ya se ejecutó, la
    aplicada.

    Ni `my_team.csv` ni el CSV de equipos distinguen a un jugador que
    entró esta semana de uno que lleva meses — solo muestran el plantel
    resultante. Sin esta sección, la transferencia solo existía en la
    salida de terminal de `run_squad.py`, que se pierde.
    """
    state = _load_base_state()
    lines: list[str] = []

    base = set(squads[squads["squad_type"] == "Base"]["web_name"])
    projected = set(squads[squads["squad_type"] == "Base proyectado"]["web_name"])
    outgoing, incoming = sorted(base - projected), sorted(projected - base)

    if outgoing or incoming:
        lines += [
            "",
            "## Transferencia propuesta para esta fecha",
            "",
            "Todavía **no está aplicada**: la decisión definitiva se toma en las últimas "
            "horas antes del deadline, cuando ya se conocen las lesiones. Esta es la "
            "propuesta con los datos de hoy, publicada para poder preparar contenido con "
            "anticipación — puede cambiar si aparece una lesión.",
            "",
            *[f"- {line}" for line in proposed_log],
            "",
            f"Sale: {', '.join(outgoing)} · Entra: {', '.join(incoming)}",
            "",
            "El equipo con el cambio ya aplicado está en `squad_recommendations.csv` "
            "bajo `squad_type = \"Base proyectado\"`.",
        ]

    applied = state.get("last_transfers")
    if applied:
        gameweek = state.get("last_transfer_gameweek", state.get("last_evaluated_gameweek"))
        lines += [
            "",
            f"## Movimientos ya ejecutados en GW{gameweek}",
            "",
            *[f"- {line}" for line in applied],
            "",
            f"Transferencias libres disponibles tras esa fecha: {state.get('banked_free_transfers', 0)}.",
        ]
    return lines


def _last_gameweek_review() -> list[str]:
    """Repaso de la última fecha cerrada: qué tan bien le pegó el modelo.

    Sale del historial de calibración, que ya cruza predicción contra
    puntos reales. Es material propio para contenido — nadie más publica
    el error de su propio modelo — y obliga a que los posts de repaso
    salgan de datos medidos y no de impresiones.
    """
    history = _load_history()
    complete = history.dropna(subset=["actual_points"])
    if complete.empty:
        return []

    last_gw = int(complete["gameweek"].max())
    gw_data = complete[complete["gameweek"] == last_gw].copy()
    gw_data["error"] = gw_data["actual_points"] - gw_data["xp_predicted"]

    lines = [
        "",
        f"## Repaso de GW{last_gw} (fecha cerrada)",
        "",
        f"Sesgo del modelo: {gw_data['error'].mean():+.2f} pts por jugador "
        f"({'subestimó' if gw_data['error'].mean() > 0 else 'sobrestimó'} en promedio). "
        f"Error absoluto medio: {gw_data['error'].abs().mean():.2f} pts.",
        "",
        "Sesgo por posición (positivo = el modelo se quedó corto):",
        "",
        "| Posición | Sesgo | Jugadores |",
        "|---|---|---|",
    ]
    for position, row in gw_data.groupby("position")["error"].agg(["mean", "count"]).iterrows():
        lines.append(f"| {position} | {row['mean']:+.2f} | {int(row['count'])} |")

    best = gw_data.nlargest(5, "actual_points")
    lines += [
        "",
        f"Mejores puntajes reales de GW{last_gw}:",
        "",
        "| Jugador | Equipo | Pos | Puntos reales | xP previsto |",
        "|---|---|---|---|---|",
    ]
    for p in best.itertuples():
        lines.append(
            f"| {p.web_name} | {p.team_name} | {p.position} | "
            f"{int(p.actual_points)} | {p.xp_predicted} |"
        )

    lines += [
        "",
        "**Ojo al usar esto:** el modelo no estima bonus points, así que "
        "subestimar es su sesgo esperado — sobre todo en mediocampistas y "
        "defensores, que son quienes más bonus reciben. Con pocas fechas "
        "acumuladas todavía es un vistazo, no una tendencia.",
    ]
    return lines


def build_briefing(players: pd.DataFrame, squads: pd.DataFrame) -> str:
    """Arma el briefing. `squads` es el DataFrame que produce
    `export_squads_for_tableau` — se reusa en vez de recalcular los
    equipos, para que el briefing no pueda contradecir al CSV.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    gameweek = int(players["next_gameweek"].dropna().mode().iloc[0])

    base_starters, base_bench = _squad_from_export(squads, "Base")
    wildcard_starters, wildcard_bench = _squad_from_export(squads, "Wildcard")

    captains = captaincy_picks(players, top_n=8)
    differentials = top_differentials(players, top_n=8)
    # top_differentials no trae techo/haul; se cruzan acá para no
    # duplicar la lógica de filtrado que ya vive en analysis.py.
    differentials = differentials.merge(
        players[["web_name", "xp_ceiling", "haul_probability", "start_rate", "set_piece_duties"]],
        on="web_name", how="left",
    )

    sections = [
        f"# The Gameweek Lab — Briefing GW{gameweek}",
        "",
        f"Generado automáticamente: {now}. Fuente única para redactar posts — "
        "ya viene filtrado y no hace falta leer los CSVs completos.",
        "",
        "## Equipo Base",
        "",
        "El equipo real, el que se sostiene fecha a fecha con transferencias normales.",
        "",
        _format_squad_table(base_starters, base_bench),
        "",
        "## Equipo Wildcard",
        "",
        "Ejercicio teórico: el mejor equipo posible si se pudiera rearmar todo hoy desde cero. "
        "Gasta lo mínimo en el banco a propósito, así que no es sostenible — sirve de comparación.",
        "",
        _format_squad_table(wildcard_starters, wildcard_bench),
        "",
        "## Mejores opciones de capitanía",
        "",
        "| Jugador | Equipo | Pos | xP | Techo | P(haul) | Propiedad | Rival |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in captains.itertuples():
        sections.append(
            f"| {c.web_name} | {c.team_name} | {c.position} | {c.xp_next} | {c.xp_ceiling} | "
            f"{c.haul_probability:.0%} | {c.selected_by_percent}% | {c.next_opponent} |"
        )

    sections += [
        "",
        "## Diferenciales (propiedad ≤ 10%)",
        "",
        "| Jugador | Equipo | Pos | Precio | Propiedad | xP | Techo | P(haul) | Rival |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in differentials.itertuples():
        sections.append(
            f"| {d.web_name} | {d.team_name} | {d.position} | £{d.now_cost}m | "
            f"{d.selected_by_percent}% | {d.xp_next} | {d.xp_ceiling} | "
            f"{d.haul_probability:.0%} | {d.next_opponent} |"
        )

    _, _, proposed_log = preview_base_transfers(players)
    sections += _transfer_section(squads, proposed_log)
    sections += _last_gameweek_review()

    sections += [
        "",
        "## Cómo leer estos números",
        "",
        f"- **xP**: puntos esperados del próximo gameweek. Ya descuenta rotación "
        f"(qué tan seguido es titular) y lesión.",
        f"- **Techo**: percentil 90 — \"en su 10% de mejores partidos saca al menos esto\". "
        "Para capitanía el promedio engaña: un arquero puede tener buen xP y 0% de haul.",
        "- **P(haul)**: probabilidad de hacer 10+ puntos.",
        f"- **xP 4GW**: puntos esperados acumulados en las próximas {HORIZON_GAMEWEEKS} fechas.",
        "- **Balón parado**: quién patea penales/tiros libres/córners. Es contexto: "
        "el xP **no** lo suma aparte, porque el xG de FPL ya incluye los penales ejecutados.",
        "",
        f"Todos los jugadores listados superan los {MIN_MINUTES_FOR_RANKING} minutos jugados. "
        "Es un filtro deliberado: por debajo de eso las tasas por 90 minutos son ruido "
        "(hay jugadores con 2 minutos jugados y el \"mejor xG90 de la liga\").",
        "",
        "El modelo no estima bonus points (BPS) ni puntos por atajadas, así que "
        "subestima levemente a delanteros y arqueros.",
        "",
    ]
    return "\n".join(sections)


def save_briefing(players: pd.DataFrame, squads: pd.DataFrame) -> str:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    briefing = build_briefing(players, squads)
    BRIEFING_PATH.write_text(briefing, encoding="utf-8")
    print(f"Guardado {BRIEFING_PATH} ({len(briefing) / 1024:.1f} KB)")
    return briefing
