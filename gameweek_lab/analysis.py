import numpy as np
import pandas as pd

from gameweek_lab.build_dataset import (
    build_players_dataset,
    get_matches_played_by_team,
    get_team_fixtures_horizon,
)
from gameweek_lab.config import DATA_PROCESSED_DIR

# Puntos por gol y por clean sheet, según posición — misma tabla que
# fpl-scoring-rules-2026-27.md, pero acá se USA para calcular, no solo
# como referencia.
GOAL_POINTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
# Puntos por jugar: 2 si arranca y completa 60+ minutos (lo asumible para
# un titular sano), 1 si juega menos. Simplificamos a 2 fijo — la
# probabilidad de jugar ya se aplica aparte, en _playing_probability.
APPEARANCE_POINTS = 2

# Debajo de esto, 'points_per_game' es ruido: pocos partidos alcanzan para
# que una sola actuación puntuda dispare el promedio sin que sea repetible.
# 900 minutos ~ 10 partidos completos.
MIN_MINUTES_FOR_RANKING = 900

# Cuántas fechas adelante mira el asesor de transferencias. Más allá de
# ~4 las proyecciones de nuestra heurística pierden sentido — la idea es
# prepararse para las próximas fechas e iterar, no predecir la temporada.
HORIZON_GAMEWEEKS = 4

# Cuánto pesa el ownership en rank_value cuando se pide una postura
# ("protect" o "chase"). El ajuste máximo posible (jugador en el extremo
# del percentil) es ±RANK_STRATEGY_WEIGHT/2 sobre el xP — con 0.6, hasta
# ±30%. Lo bastante fuerte para reordenar el ranking entre opciones
# parecidas, sin que un diferencial mediocre le gane a un template
# excelente solo por ser poco poseído.
RANK_STRATEGY_WEIGHT = 0.6
RANK_STANCES = ("protect", "neutral", "chase")

# Parámetros del factor de rotación (ver `_start_rate`).
FULL_SEASON_MATCHES = 38
# Hacia dónde tira la tasa de titularidad cuando hay poca evidencia. 0.75
# ≈ "titular habitual pero no indiscutido", razonable para el pool que ya
# pasó el filtro de MIN_MINUTES_FOR_RANKING.
START_RATE_PRIOR = 0.75
# Cuántos partidos de evidencia "vale" el prior. Con 5: en GW2 (1 partido
# jugado) la tasa apenas se mueve del prior, y recién con ~15 fechas los
# datos de la temporada en curso dominan. Evita que 2 de 2 se lea como
# "100% titular", o 0 de 2 como "nunca juega".
START_RATE_PRIOR_WEIGHT = 5


def _base_scoring_rate(players: pd.DataFrame) -> pd.Series:
    """Puntos esperados 'crudos' por 90 minutos, antes de ajustar por rival
    — calculados desde estadísticas subyacentes (goles/asistencias
    esperados, goles esperados en contra), no desde puntos ya anotados.

    Antes usábamos 'form'/'points_per_game': puntos reales, que mezclan la
    calidad del jugador con la suerte de un partido puntual (un bonus de
    3 puntos por estar en el lugar correcto no dice nada sobre si va a
    repetirse). xG/xA reflejan las oportunidades que el jugador genera o
    convierte en el proceso, no el resultado puntual — un proxy más
    estable de la habilidad real.

    Clean sheet: se estima P(0 goles en contra) con Poisson —
    exp(-goles_esperados_en_contra_por_90) — un supuesto estándar en
    analítica de fútbol (los goles en un partido se aproximan bien a una
    distribución Poisson).

    Limitación conocida: no modela bonus points (BPS). FPL no expone un
    "bono esperado" por jugador, y aproximarlo requeriría replicar las
    ~32 métricas del sistema de BPS. Para delanteros/mediocampistas de
    ataque, el bono suele sumar un 15-20% extra sobre sus puntos reales
    — este modelo los subestima un poco de forma sistemática. Tampoco
    modela puntos por atajadas de arqueros.
    """
    goal_points = players["position"].map(GOAL_POINTS)
    clean_sheet_points = players["position"].map(CLEAN_SHEET_POINTS)

    attacking = (
        players["expected_goals_per_90"] * goal_points
        + players["expected_assists_per_90"] * ASSIST_POINTS
    )
    clean_sheet_probability = np.exp(-players["expected_goals_conceded_per_90"])
    defensive = clean_sheet_probability * clean_sheet_points

    return attacking + defensive + APPEARANCE_POINTS


def _fixture_multiplier(difficulty: pd.Series) -> pd.Series:
    """Convierte el FDR de FPL (1=fácil, 5=difícil) en un multiplicador.

    Centrado en dificultad 3 (multiplicador 1.0): rivales más fáciles suben
    el xP, más difíciles lo bajan. Rango aprox. 0.33x (rival top) a 1.67x
    (rival más débil).
    """
    return (6 - difficulty) / 3


def _playing_probability(players: pd.DataFrame) -> pd.Series:
    """Probabilidad de LESIÓN/suspensión únicamente, no de rotación.

    `chance_of_playing_next_round` viene en % (0-100), o NaN cuando el
    club no reportó ninguna duda física — ahí asumimos 100%. Ojo con la
    interpretación: verificado contra datos reales, solo 9 de 224
    jugadores elegibles tienen valor no nulo, así que este factor es un
    no-op para el ~96% del pool. La rotación se modela aparte, en
    `_start_rate` — son cosas distintas y este campo no la captura.
    """
    return players["chance_of_playing_next_round"].fillna(100) / 100


def _start_rate(players: pd.DataFrame) -> pd.Series:
    """Con qué frecuencia el jugador es TITULAR — el factor de rotación,
    independiente de la duda por lesión (`_playing_probability`).

    Por qué hace falta: `chance_of_playing_next_round` solo se llena
    cuando hay una lesión reportada, así que un suplente habitual
    perfectamente sano figura con "100% de probabilidad de jugar". Sin
    esto, el modelo trataba igual a un titular indiscutido y a alguien
    que entra 20 minutos: verificado en datos reales, Nmecha (10 starts)
    y Thiago (37 starts) tenían el mismo factor de disponibilidad.

    Se usa `starts` (no `minutes`) porque distingue al titular del
    suplente que suma minutos entrando: dos jugadores con los mismos
    minutos totales pueden ser uno titular fijo y otro rotativo.

    Suavizado (shrinkage) hacia `START_RATE_PRIOR`: con pocos partidos
    jugados, la tasa cruda es ruido — 2 de 2 no significa "100% titular".
    `START_RATE_PRIOR_WEIGHT` equivale a partidos de evidencia previa,
    así que la tasa arranca en el prior y se va acercando a la real a
    medida que se acumulan partidos.
    """
    played = get_matches_played_by_team()
    team_matches = players["team_name"].map(played).fillna(0)
    # Pre-temporada no hay partidos jugados todavía: `starts` es de la
    # temporada anterior, así que el denominador correcto es esa temporada
    # completa, no cero.
    team_matches = team_matches.where(team_matches > 0, FULL_SEASON_MATCHES)

    starts = players["starts"].fillna(0)
    return (
        (starts + START_RATE_PRIOR_WEIGHT * START_RATE_PRIOR)
        / (team_matches + START_RATE_PRIOR_WEIGHT)
    ).clip(upper=1.0)


def _availability(players: pd.DataFrame) -> pd.Series:
    """Probabilidad de que el jugador aporte puntos en la próxima fecha:
    rotación × lesión. Las dos son independientes y se modelan aparte
    (ver `_start_rate` y `_playing_probability`).
    """
    return _start_rate(players) * _playing_probability(players)


def add_expected_points(players: pd.DataFrame) -> pd.DataFrame:
    players = players.copy()
    base_rate = _base_scoring_rate(players)
    fixture_mult = _fixture_multiplier(players["next_fixture_difficulty"])

    # `start_rate` se expone como columna propia (no solo multiplicada
    # adentro del xP) para poder explicar POR QUÉ un jugador con buenas
    # estadísticas igual proyecta bajo: no es que rinda mal, es que no
    # arranca seguido.
    players["start_rate"] = _start_rate(players).round(2)
    players["xp_next"] = (base_rate * fixture_mult * _availability(players)).round(2)
    return players


def add_horizon_expected_points(players: pd.DataFrame, horizon: int = HORIZON_GAMEWEEKS) -> pd.DataFrame:
    """Agrega xp_horizon: puntos esperados acumulados en las próximas
    `horizon` fechas, sumando el multiplicador de dificultad de cada
    partido del equipo (double gameweeks suman dos partidos, blanks cero).

    Disponibilidad: la rotación (`_start_rate`) aplica a TODAS las fechas
    del horizonte — un suplente habitual lo sigue siendo dentro de 4
    semanas. La duda por lesión, en cambio, se trata distinto según la
    fecha: para la primera usamos `chance_of_playing_next_round` tal
    cual; para las siguientes asumimos que un jugador sano ('a') se
    recupera y juega normal, pero que la duda de un lesionado/suspendido
    persiste — conservador a propósito, un asesor de transferencias DEBE
    penalizar tener jugadores que quizás no jueguen.
    """
    players = players.copy()
    fixtures = get_team_fixtures_horizon(horizon)
    fixtures["multiplier"] = _fixture_multiplier(fixtures["difficulty"])
    fixtures["label"] = fixtures.apply(
        lambda r: f"{r['opponent']}({'H' if r['is_home'] else 'A'})", axis=1
    )

    first_gw = fixtures["gameweek"].min()
    is_first = fixtures["gameweek"] == first_gw
    mult_first = fixtures[is_first].groupby("team_name")["multiplier"].sum()
    mult_rest = fixtures[~is_first].groupby("team_name")["multiplier"].sum()
    summary = fixtures.sort_values("gameweek").groupby("team_name")["label"].agg(" · ".join)

    base_rate = _base_scoring_rate(players)
    start_rate = _start_rate(players)
    prob_next = start_rate * _playing_probability(players)
    later_availability = start_rate * (
        players["chance_of_playing_next_round"].fillna(0).div(100)
        .where(players["status"] != "a", 1.0)
    )

    players["xp_horizon"] = (
        base_rate
        * (
            players["team_name"].map(mult_first).fillna(0.0) * prob_next
            + players["team_name"].map(mult_rest).fillna(0.0) * later_availability
        )
    ).round(2)
    players["fixtures_horizon"] = players["team_name"].map(summary).fillna("—")
    return players


def add_rank_adjusted_value(players: pd.DataFrame, stance: str = "neutral", xp_column: str = "xp_next") -> pd.DataFrame:
    """Agrega `rank_value`: el xP visto a través de una postura de rank.

    El xP puro (`xp_next`/`xp_horizon`) es una estimación de valor
    esperado, sin opinión sobre riesgo — no distingue entre "proteger una
    buena posición en tu liga" (conviene baja varianza: jugadores de alto
    ownership, así si fallan, le fallan a todos tus rivales por igual) y
    "remontar desde atrás" (conviene alta varianza: diferenciales de bajo
    ownership, que si aciertan te separan del resto — un template no te
    separa de nadie aunque rinda).

    - `stance="neutral"` (default): rank_value = xp, sin ajuste — así
      queda todo lo ya construido (equipo Base, Wildcard) sin cambios de
      comportamiento salvo que se pida explícitamente lo contrario.
    - `stance="protect"`: favorece ownership alto.
    - `stance="chase"`: favorece ownership bajo (diferenciales).

    El ownership se compara por PERCENTIL dentro del propio `players`
    recibido, no por un umbral fijo — la distribución real está muy
    sesgada (mediana ~1.6% entre candidatos con muestra confiable, media
    ~5.5%, algún template en 70%), así que un pivote fijo tipo "50%" no
    tendría sentido; el percentil se autocalibra al pool que se le pase.
    """
    players = players.copy()
    if stance not in RANK_STANCES:
        raise ValueError(f"stance debe ser uno de {RANK_STANCES}, recibí '{stance}'")

    if stance == "neutral":
        players["rank_value"] = players[xp_column]
        return players

    tilt = 1 if stance == "chase" else -1
    ownership_percentile = players["selected_by_percent"].rank(pct=True)
    ownership_edge = 0.5 - ownership_percentile  # +0.5 = el menos poseído, -0.5 = el más poseído
    players["rank_value"] = (
        players[xp_column] * (1 + tilt * RANK_STRATEGY_WEIGHT * ownership_edge)
    ).round(2)
    return players


def top_differentials(players: pd.DataFrame, ownership_max: float = 10.0, top_n: int = 10) -> pd.DataFrame:
    """Buenos jugadores que casi nadie tiene: baja propiedad + buen xP.

    Excluye a los de pocos minutos: con muestra chica, 'buen xP' suele ser
    ruido (un partido puntudo aislado) y no una señal real de rendimiento.
    """
    candidates = players[
        (players["selected_by_percent"] <= ownership_max)
        & (players["minutes"] >= MIN_MINUTES_FOR_RANKING)
    ]
    columns = ["web_name", "team_name", "position", "now_cost", "selected_by_percent", "xp_next", "next_opponent"]
    return candidates.sort_values("xp_next", ascending=False).head(top_n)[columns]


def captaincy_picks(players: pd.DataFrame, top_n: int = 5, stance: str = "neutral") -> pd.DataFrame:
    """Mejores opciones de capitán para el próximo gameweek: alto xP, buena
    probabilidad de jugar, y minutos suficientes para confiar en el dato.

    La capitanía es donde más pesa la varianza (duplica el puntaje) — con
    `stance="protect"` prioriza capitanes de alto ownership (si falla, le
    falla a todos tus rivales); con `stance="chase"`, diferenciales que
    puedan separarte del resto de tu liga. Ver add_rank_adjusted_value.
    """
    likely_to_play = players[
        (players["chance_of_playing_next_round"].fillna(100) >= 75)
        & (players["minutes"] >= MIN_MINUTES_FOR_RANKING)
    ]
    likely_to_play = add_rank_adjusted_value(likely_to_play, stance)
    columns = ["web_name", "team_name", "now_cost", "xp_next", "selected_by_percent", "rank_value", "next_opponent", "next_is_home"]
    return likely_to_play.sort_values("rank_value", ascending=False).head(top_n)[columns]


def save_scored_players(players: pd.DataFrame) -> None:
    """Guarda el pool completo de jugadores con xP calculado — fuente de
    datos para dashboards externos (Tableau, Power BI, lo que sea) sobre
    todo el pool, no solo el equipo recomendado: diferenciales, forma vs
    precio, capitanía, lo que se quiera cruzar.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "players_scored.csv"
    players.to_csv(out_path, index=False)
    print(f"Guardado {out_path} ({len(players)} jugadores)")


def run_analysis() -> None:
    players = build_players_dataset()
    players = add_expected_points(players)
    save_scored_players(players)

    print("\n=== Top 5 opciones de capitanía — próximo gameweek ===")
    print(captaincy_picks(players).to_string(index=False))

    print("\n=== Top 10 diferenciales (ownership <= 10%) ===")
    print(top_differentials(players).to_string(index=False))


if __name__ == "__main__":
    run_analysis()
