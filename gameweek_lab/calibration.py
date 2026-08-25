"""Loop de calibración: compara el xP proyectado contra los puntos reales,
fecha a fecha, para poder responder con datos si el modelo tiene sesgos
sistemáticos (¿sobrestima delanteros? ¿subestima defensas baratas?) — en
vez de solo confiar en que la heurística "se siente bien".

Dos pasos independientes, cada uno gateado para no duplicar trabajo:

1. `snapshot_predictions()`: antes de que se juegue cada fecha, graba qué
   proyectó el modelo para cada jugador. Sin esto la proyección se
   pierde apenas se refrescan los datos al día siguiente —
   `players_scored.csv` se sobreescribe, no guarda historial.
2. `record_actual_points()`: una vez que una fecha queda con resultados y
   bonus points ya asignados (todos sus partidos terminados),
   completa los puntos reales de esa fecha en el historial, usando
   `/event/{id}/live/` — a diferencia de `event_points` en
   bootstrap-static (que solo refleja la fecha "actual" del juego y se
   pisa apenas arranca la siguiente), este endpoint no es ambiguo sobre
   a qué fecha se refiere.

Deliberadamente NO auto-corrige el modelo con lo que encuentra: con 0
fechas jugadas todavía (la temporada 2026/27 arranca el 21/8),
"corregir" ahora sería ajustar contra ruido, no señal. `build_calibration_report`
imprime el análisis para que lo lea una persona — decidir si vale ajustar
el modelo, una vez haya semanas suficientes, es aparte.
"""

import json

import pandas as pd

from gameweek_lab.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from gameweek_lab.fetch import fetch_event_live

CALIBRATION_PATH = DATA_PROCESSED_DIR / "xp_calibration.csv"
CALIBRATION_COLUMNS = [
    "gameweek", "player_id", "web_name", "position", "team_name",
    "xp_predicted", "now_cost_at_prediction", "selected_by_percent_at_prediction",
    "actual_points",
]
# El umbral se cuenta en FECHAS, no en observaciones. Una fecha aporta
# ~480 filas, pero no son 480 evidencias independientes: comparten los
# mismos 10 partidos, el mismo clima de resultados y las mismas sorpresas.
# Un umbral por observaciones daba vía libre tras un solo gameweek, que es
# justo cuando las conclusiones son menos confiables.
MIN_GAMEWEEKS_FOR_BIAS_REPORT = 4


def _load_history() -> pd.DataFrame:
    if CALIBRATION_PATH.exists():
        return pd.read_csv(CALIBRATION_PATH)
    return pd.DataFrame(columns=CALIBRATION_COLUMNS)


def _save_history(history: pd.DataFrame) -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(CALIBRATION_PATH, index=False)


def _completed_gameweeks() -> set[int]:
    """Gameweeks cuyos partidos ya terminaron todos.

    Se mira `finished_provisional` (o `finished`) en los fixtures, no
    `data_checked` en los events. Verificado en GW1: los 10 partidos
    terminaron, los bonus ya estaban asignados y los puntos eran
    definitivos, pero `data_checked` seguía en False — FPL lo marca
    recién tras su verificación final, que puede tardar días. Esperar ese
    flag dejaba la calibración parada indefinidamente sobre datos que ya
    estaban completos.

    Exige que TODOS los partidos de la fecha hayan terminado: capturar con
    un partido en curso daría puntos parciales.
    """
    path = DATA_RAW_DIR / "fixtures.json"
    with open(path, encoding="utf-8") as f:
        fixtures = json.load(f)

    by_gameweek: dict[int, list[bool]] = {}
    for fixture in fixtures:
        gameweek = fixture.get("event")
        if gameweek is None:
            continue  # partido postergado sin fecha asignada
        done = bool(fixture.get("finished") or fixture.get("finished_provisional"))
        by_gameweek.setdefault(gameweek, []).append(done)

    return {gw for gw, done_flags in by_gameweek.items() if all(done_flags)}


def snapshot_predictions(players: pd.DataFrame) -> str:
    """Graba xp_next de cada jugador disponible para el próximo
    gameweek. Una sola vez por fecha — si ya se grabó, no hace nada
    (evita duplicar filas si el pipeline corre varias veces el mismo día).
    """
    history = _load_history()
    next_gw = int(players["next_gameweek"].dropna().mode().iloc[0])

    if not history.empty and (history["gameweek"] == next_gw).any():
        return f"GW{next_gw} ya tiene predicción grabada — sin cambios."

    eligible = players[players["status"] == "a"].copy()
    snapshot = pd.DataFrame({
        "gameweek": next_gw,
        "player_id": eligible["id"],
        "web_name": eligible["web_name"],
        "position": eligible["position"],
        "team_name": eligible["team_name"],
        "xp_predicted": eligible["xp_next"],
        "now_cost_at_prediction": eligible["now_cost"],
        "selected_by_percent_at_prediction": eligible["selected_by_percent"],
        "actual_points": pd.NA,
    })
    history = pd.concat([history, snapshot], ignore_index=True)
    _save_history(history)
    return f"GW{next_gw}: predicción grabada para {len(snapshot)} jugadores."


def record_actual_points() -> str:
    """Completa `actual_points` para cualquier fecha ya finalizada
    (todos sus partidos jugados) que todavía tenga huecos en el historial.

    Frágil a propósito, documentado: si el pipeline no corre por un buen
    tiempo, no se pierde nada — /event/{id}/live/ siempre da los puntos
    de esa fecha puntual sin importar cuánto avanzó la temporada después
    (a diferencia de event_points en bootstrap-static).
    """
    history = _load_history()
    if history.empty:
        return "Sin predicciones grabadas todavía — nada que completar."

    pending_mask = history["actual_points"].isna()
    if not pending_mask.any():
        return "El historial ya tiene todos los puntos reales completados."

    completed_gws = _completed_gameweeks()
    pending_gws = sorted(set(history.loc[pending_mask, "gameweek"]) & completed_gws)

    if not pending_gws:
        return "No hay fechas finalizadas todavía pendientes de completar."

    filled = 0
    for gw in pending_gws:
        live = fetch_event_live(gw)
        if not live["elements"]:
            continue  # la fecha figura terminada pero el endpoint vino vacío — se reintenta después
        points_by_id = {e["id"]: e["stats"]["total_points"] for e in live["elements"]}
        rows = history.index[(history["gameweek"] == gw) & pending_mask]
        for idx in rows:
            player_id = history.loc[idx, "player_id"]
            if player_id in points_by_id:
                history.loc[idx, "actual_points"] = points_by_id[player_id]
                filled += 1

    _save_history(history)
    return f"Completados {filled} puntos reales en {len(pending_gws)} fecha(s): {pending_gws}."


def build_calibration_report() -> str:
    """Compara xP proyectado vs. puntos reales — sesgo general y por
    posición. Con pocas observaciones, dice explícitamente que es
    demasiado pronto para sacar conclusiones, en vez de mostrar un
    "sesgo" que en realidad es ruido de muestra chica.
    """
    history = _load_history()
    complete = history.dropna(subset=["actual_points"])
    if complete.empty:
        return "Sin puntos reales registrados todavía — nada que calibrar."

    complete = complete.copy()
    complete["error"] = complete["actual_points"] - complete["xp_predicted"]

    lines = [
        f"=== Calibración del modelo — {len(complete)} observaciones "
        f"({complete['gameweek'].nunique()} fecha(s)) ==="
    ]
    gameweeks = complete["gameweek"].nunique()
    if gameweeks < MIN_GAMEWEEKS_FOR_BIAS_REPORT:
        lines.append(
            f"AVISO: solo {gameweeks} fecha(s) de datos (hacen falta "
            f"{MIN_GAMEWEEKS_FOR_BIAS_REPORT} para leer esto como tendencia). Las "
            f"{len(complete)} filas provienen de los mismos partidos, así que no son "
            "observaciones independientes: es un vistazo, no un diagnóstico."
        )

    overall_bias = complete["error"].mean()
    mae = complete["error"].abs().mean()
    direction = "el modelo SUBestima" if overall_bias > 0 else "el modelo SOBREestima"
    lines.append(
        f"Sesgo general: {overall_bias:+.2f} pts/jugador ({direction} en promedio) "
        f"| Error absoluto promedio: {mae:.2f} pts"
    )

    by_position = complete.groupby("position")["error"].agg(["mean", "count"]).round(2)
    lines.append("\nSesgo por posición (positivo = subestima, negativo = sobrestima):")
    lines.append(by_position.to_string())

    return "\n".join(lines)
