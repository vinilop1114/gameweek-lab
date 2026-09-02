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

Deliberadamente NO auto-corrige el modelo con lo que encuentra: con dos
o tres fechas jugadas, "corregir" sería ajustar contra ruido, no señal.
`build_calibration_report` imprime el análisis para que lo lea una
persona — decidir si vale ajustar el modelo, una vez haya semanas
suficientes, es aparte.
"""

import json

import pandas as pd

from gameweek_lab.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from gameweek_lab.fetch import fetch_event_live

CALIBRATION_PATH = DATA_PROCESSED_DIR / "xp_calibration.csv"
CALIBRATION_COLUMNS = [
    "gameweek", "player_id", "web_name", "position", "team_name",
    "xp_predicted", "now_cost_at_prediction", "selected_by_percent_at_prediction",
    "points_per_game_at_prediction", "ep_next_at_prediction", "start_rate_at_prediction",
    "actual_points",
]
# Predictores contra los que se compara el modelo, además de él mismo.
# Sin un baseline, un MAE de 1.44 no dice nada: lo que importa es si
# `xp_next` ordena mejor que alternativas triviales que no requieren
# modelo alguno. `ep_next` es la propia estimación de FPL, el rival
# natural; `points_per_game` es el predictor de una línea; y
# `selected_by_percent` es el consenso del mercado.
BASELINE_COLUMNS = {
    "xp_predicted": "xp_next (el modelo)",
    "ep_next_at_prediction": "ep_next (estimación de FPL)",
    "points_per_game_at_prediction": "points_per_game (temporada previa)",
    "selected_by_percent_at_prediction": "selected_by_percent (el mercado)",
    "start_rate_at_prediction": "start_rate (solo titularidad)",
    "now_cost_at_prediction": "now_cost (solo precio)",
}
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
        # Baselines congelados junto con la predicción: comparar contra
        # ellos más tarde exige tener su valor *previo al deadline*, no el
        # de hoy. Sin esto no hay forma de saber si el modelo aporta algo
        # sobre predictores triviales.
        "points_per_game_at_prediction": eligible["points_per_game"],
        "ep_next_at_prediction": eligible["ep_next"],
        "start_rate_at_prediction": eligible["start_rate"],
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

    lines += _availability_vs_scoring_section(complete)
    lines += _ranking_power_section(complete)

    played = complete[complete["actual_points"] > 0]
    by_position = played.groupby("position")["error"].agg(["mean", "count"]).round(2)
    lines.append("\n-- Sesgo por posición (SOLO jugadores con puntos) --")
    lines.append("Positivo = el modelo se quedó corto.")
    lines.append(by_position.to_string())

    return "\n".join(lines)


def _availability_vs_scoring_section(complete: pd.DataFrame) -> list[str]:
    """Separa el error de disponibilidad del error de scoring.

    El sesgo agregado promedia dos poblaciones que fallan en direcciones
    opuestas y se cancelan: al que no juega se le regalan puntos que nunca
    va a cobrar (sesgo negativo), y al que juega se le queda corto (sesgo
    positivo). Un solo número no permite saber cuál de los dos problemas
    se está arreglando.

    Cuidado al leer la fila "jugó": una parte de ese sesgo es artefacto de
    condicionar, no un defecto. Si a un jugador con 60% de probabilidad de
    arrancar se le cobran 1.2 de los 2 puntos de aparición, al mirar solo
    a quienes efectivamente jugaron el modelo SIEMPRE va a quedar corto.
    Ese componente es correcto por diseño (el xP es un valor esperado
    incondicional); lo que importa es cuánto sesgo queda por encima de él.
    """
    played = complete[complete["actual_points"] > 0]
    missed = complete[complete["actual_points"] == 0]

    rows = [
        ("No apareció", missed["error"].mean(), len(missed)),
        ("Jugó", played["error"].mean(), len(played)),
        ("Agregado", complete["error"].mean(), len(complete)),
    ]
    lines = ["\n-- Sesgo separado: disponibilidad vs scoring --"]
    lines.append(f"{'Población':<14}{'Sesgo':>9}{'n':>7}")
    for label, bias, n in rows:
        bias_text = f"{bias:+.2f}" if n else "—"
        lines.append(f"{label:<14}{bias_text:>9}{n:>7}")

    if len(played):
        # Cuánto del sesgo de los que jugaron es el artefacto de
        # condicionar: los puntos de aparición que el modelo repartió
        # hacia el escenario "no juega" y que, visto solo entre quienes
        # jugaron, nunca podía cobrar.
        start_rate = complete.get("start_rate_at_prediction")
        if start_rate is not None and played["start_rate_at_prediction"].notna().any():
            expected_shortfall = (2 * (1 - played["start_rate_at_prediction"])).mean()
            residual = played["error"].mean() - expected_shortfall
            lines.append(
                f"\nDe los {played['error'].mean():+.2f} de quienes jugaron, ~{expected_shortfall:.2f} "
                f"es artefacto de condicionar (puntos de aparición asignados al escenario "
                f"'no juega'). Sesgo de scoring residual: {residual:+.2f}."
            )
        else:
            lines.append(
                "\n(Sin `start_rate_at_prediction` en estas fechas no se puede separar el "
                "artefacto de condicionar del sesgo real; disponible desde el próximo snapshot.)"
            )
    return lines


def _ranking_power_section(complete: pd.DataFrame) -> list[str]:
    """Poder de ORDENAMIENTO del modelo frente a predictores triviales.

    El sesgo es una constante que se puede corregir sumando; lo que el
    modelo realmente vende es el ranking — a quién capitanear, a quién
    transferir. Un modelo perfectamente insesgado puede ser inútil para
    elegir jugadores, y uno sesgado puede ordenar perfecto.

    Se usa Spearman (correlación de rangos) porque mide exactamente eso:
    si el orden propuesto coincide con el orden real, sin importar la
    escala. Si `xp_next` no le gana a `points_per_game` de forma
    sostenida, la complejidad del modelo no se está pagando.
    """
    lines = ["\n-- Poder de ordenamiento (Spearman vs puntos reales) --"]
    available = {
        column: label for column, label in BASELINE_COLUMNS.items()
        if column in complete.columns and complete[column].notna().any()
    }
    if len(available) <= 1:
        lines.append(
            "Todavía sin baselines guardados para comparar — se registran desde el "
            "próximo snapshot (points_per_game, ep_next, start_rate)."
        )
        return lines

    scores = []
    for column, label in available.items():
        subset = complete[[column, "actual_points"]].dropna()
        if len(subset) > 10:
            # Spearman = Pearson sobre los rangos. Se calcula así en vez
            # de con method="spearman" porque esa vía exige scipy, una
            # dependencia pesada para una transformación de una línea
            # (mismo criterio que la Poisson en analysis.py).
            score = subset[column].rank().corr(subset["actual_points"].rank())
            scores.append((label, score))

    lines.append(f"{'Predictor':<38}{'Spearman':>10}")
    for label, score in sorted(scores, key=lambda item: item[1], reverse=True):
        marker = "  <-- el modelo" if label.startswith("xp_next") else ""
        lines.append(f"{label:<38}{score:>10.3f}{marker}")
    lines.append(
        "\nSi un predictor trivial le gana al modelo de forma sostenida, la complejidad "
        "no se está pagando. Una fecha aislada no alcanza para concluirlo."
    )
    return lines
