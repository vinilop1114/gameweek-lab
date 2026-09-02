import math

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
# Métricas de la temporada anterior, congeladas en pre-temporada. FPL
# resetea TODOS los acumulados al empezar una temporada nueva (minutos,
# starts, goles esperados...). Sin esta referencia, el modelo se quedaría
# sin insumo: en GW2, el xG90 de un jugador saldría de un solo partido.
LAST_SEASON_BASELINE_PATH = DATA_PROCESSED_DIR / "last_season_baseline.csv"
# Métricas que se congelan. Son tasas por 90' (comparables entre
# temporadas) más `starts`/`minutes`, que dan el peso de la evidencia.
BASELINE_RATE_COLUMNS = [
    "expected_goals_per_90",
    "expected_assists_per_90",
    "expected_goals_conceded_per_90",
]
# Cuántos minutos de la temporada en curso hacen falta para que los datos
# nuevos pesen lo mismo que los de la anterior. 900 (~10 partidos) es el
# mismo umbral que MIN_MINUTES_FOR_RANKING usa para considerar una muestra
# confiable: antes de eso, la temporada pasada sigue siendo mejor evidencia.
BASELINE_BLEND_MINUTES = 900
# Minutos mínimos en la temporada anterior para que su tasa de titularidad
# sirva como prior. Con 0 minutos, un `start_rate` de 0 no significa "era
# suplente" sino "no jugó en esta liga" — fichaje, ascenso o cesión — y
# usarlo como prior hunde a jugadores que hoy son titulares.
#
# El umbral es 1 y no un número mayor a propósito: 200 de los 600
# jugadores del baseline tienen exactamente 0 minutos (caso inequívoco de
# ausencia de información), mientras que alguien con 300 minutos sí dice
# algo sobre su rol. Poner el corte más arriba sería inventar un umbral
# sin evidencia y descartaría señal real de casi la mitad del pool.
MIN_BASELINE_MINUTES = 1

# Contribución defensiva (DEFCON): 2 puntos por partido al superar un
# umbral de acciones defensivas. Los umbrales son los oficiales de
# 2026/27 (ver initial setup/fpl-scoring-rules): defensores necesitan 10
# CBIT (despejes, bloqueos, intercepciones, entradas); medios y
# delanteros 12 CBIRT (lo mismo más recuperaciones). Los arqueros no
# reciben DEFCON.
DEFCON_THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12}
DEFCON_POINTS = 2
# Minutos de la temporada en curso para que la tasa defensiva propia pese
# igual que la mediana de su posición. El baseline de pre-temporada no
# guardó esta métrica (se agregó después de congelarlo), así que no hay
# datos del año pasado con qué mezclar: el shrinkage va contra la mediana
# posicional hasta que se acumulen minutos.
#
# Es menor que BASELINE_BLEND_MINUTES (900) a propósito: son 10-20
# acciones por partido, no 0-1 goles, así que la tasa se estabiliza con
# muchos menos minutos que el xG. Exigir la misma muestra que para goles
# dejaría a todo el pool pegado a la mediana durante media temporada.
DEFCON_BLEND_MINUTES = 270

# Parámetros de la distribución de puntos (ver `_points_distribution`).
# Los máximos son cortes prácticos. Para un jugador típico (xG90 ~0.5)
# P(4+ goles) ya es despreciable, pero se toma margen porque hay tasas por
# 90' infladas por muestra chica (algún jugador con pocos minutos llega a
# xG90 > 3). Igual la pmf se renormaliza, así que el corte no rompe nada.
MAX_GOALS = 10
MAX_ASSISTS = 6
CEILING_PERCENTILE = 0.90
# 10 puntos es el umbral clásico de "haul" en FPL (gol + clean sheet +
# bonus, o doblete).
HAUL_THRESHOLD = 10


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

    # Las tasas se mezclan con las de la temporada anterior según cuántos
    # minutos lleve jugados el jugador (ver `_blend_with_baseline`). FPL
    # resetea estos acumulados cada temporada, así que sin esto el modelo
    # se quedaría sin insumo en las primeras fechas.
    expected_goals = _blend_with_baseline(players, "expected_goals_per_90")
    expected_assists = _blend_with_baseline(players, "expected_assists_per_90")
    expected_conceded = _blend_with_baseline(players, "expected_goals_conceded_per_90")

    attacking = expected_goals * goal_points + expected_assists * ASSIST_POINTS
    clean_sheet_probability = np.exp(-expected_conceded)
    defensive = clean_sheet_probability * clean_sheet_points

    return attacking + defensive + _defcon_expected_points(players) + APPEARANCE_POINTS


def _defcon_expected_points(players: pd.DataFrame) -> pd.Series:
    """Puntos esperados por contribución defensiva.

    DEFCON paga 2 puntos fijos al superar un umbral de acciones
    defensivas en el partido, así que lo que hay que estimar es la
    PROBABILIDAD de superarlo, no un promedio. Se modela el conteo de
    acciones como Poisson con λ = tasa por 90' del jugador — el mismo
    supuesto que ya se usa para goles.

    La tasa propia se mezcla con la mediana de su posición ponderando por
    minutos jugados: el baseline de pre-temporada no guardó esta métrica
    (se agregó después de congelarlo), así que no hay datos del año
    anterior con los cuales estabilizar a quien lleva pocos minutos.

    Limitación: Poisson asume varianza igual a la media, y las acciones
    defensivas son más consistentes partido a partido que los goles. Eso
    infla levemente la probabilidad de los umbrales lejanos. Es
    preferible a ignorar el componente por completo, y la calibración lo
    va a mostrar si el sesgo es material.
    """
    thresholds = players["position"].map(DEFCON_THRESHOLDS)
    rate = players["defensive_contribution_per_90"].fillna(0)
    minutes = players["minutes"].fillna(0)

    # La mediana se calcula SOLO sobre quienes jugaron: 265 de 629
    # jugadores llevan 0 minutos y su tasa de 0 hundía la referencia
    # (DEF caía de 7.0 a 4.5), castigando de más a todo el mundo.
    played = players[minutes > 0]
    medians = played.groupby("position")["defensive_contribution_per_90"].median()
    position_median = players["position"].map(medians).fillna(0)
    blended = (rate * minutes + position_median * DEFCON_BLEND_MINUTES) / (minutes + DEFCON_BLEND_MINUTES)

    # P(acciones >= umbral) = 1 - P(acciones <= umbral - 1)
    max_threshold = int(thresholds.max()) if thresholds.notna().any() else 0
    pmf = _poisson_pmf_matrix(blended.to_numpy(), max_threshold)
    cumulative = pmf.cumsum(axis=1)

    reaches = np.zeros(len(players))
    for position, threshold in DEFCON_THRESHOLDS.items():
        mask = (players["position"] == position).to_numpy()
        reaches[mask] = 1 - cumulative[mask, threshold - 1]
    return pd.Series(reaches * DEFCON_POINTS, index=players.index)


def _poisson_pmf_matrix(lambdas: np.ndarray, max_k: int) -> np.ndarray:
    """P(exactamente k eventos) para k = 0..max_k, vectorizado.

    Devuelve una matriz (n_jugadores, max_k + 1). Se calcula a mano
    (`exp(-λ)·λ^k / k!`) en vez de usar scipy para no sumar una
    dependencia pesada por una fórmula de una línea.
    """
    k = np.arange(max_k + 1)
    factorials = np.array([math.factorial(int(i)) for i in k], dtype=float)
    lam = np.asarray(lambdas, dtype=float)[:, None]
    pmf = np.exp(-lam) * lam**k / factorials
    # Truncar en max_k deja fuera la cola, así que las probabilidades no
    # sumarían 1 y el percentil se calcularía sobre una distribución
    # incompleta. Se renormaliza para redistribuir esa masa: robusto ante
    # cualquier λ, incluso los inflados por muestra chica (hay jugadores
    # con xG90 > 3 por haber jugado pocos minutos).
    return pmf / pmf.sum(axis=1, keepdims=True)


def _points_distribution(players: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Distribución completa de puntos por jugador para la próxima fecha,
    no solo su promedio.

    El xP lineal responde "cuánto saca en promedio", pero para capitanía
    importa el techo: duplicar a un jugador de 5 xP consistente no es lo
    mismo que duplicar a uno de 5 xP que reparte entre blanks y hauls. Un
    delantero con λ alto tiene una cola derecha mucho más gorda (chance
    real de doblete o triplete) que un defensor con el mismo promedio.

    Modelo: goles ~ Poisson(xG90 · fixture_mult), asistencias ~
    Poisson(xA90 · fixture_mult), clean sheet ~ Bernoulli, los tres
    independientes entre sí. Se enumeran todas las combinaciones y sus
    probabilidades. Con `1 - availability` el jugador no juega y saca 0
    (no arranca o está lesionado) — ese riesgo es parte del techo real de
    una decisión de capitanía.

    Devuelve (puntos, probabilidades), ambas matrices (n_jugadores,
    n_resultados), alineadas fila a fila con `players`.

    Aproximación conocida: el `fixture_multiplier` escala los λ y la
    probabilidad de clean sheet, pero no los 2 puntos de aparición (que
    no dependen del rival). El xP lineal sí los escala, así que la media
    de esta distribución no coincide exactamente con `xp_next` — difieren
    en ese término. Se mantiene así para no alterar `xp_next`, que ya
    está en uso y grabado en el historial de calibración.
    """
    fixture_mult = _fixture_multiplier(players["next_fixture_difficulty"]).fillna(0).to_numpy()
    goal_points = players["position"].map(GOAL_POINTS).to_numpy()
    clean_sheet_points = players["position"].map(CLEAN_SHEET_POINTS).to_numpy()

    # Mismas tasas mezcladas que usa `_base_scoring_rate`, para que la
    # distribución y el promedio no se calculen sobre insumos distintos.
    goal_pmf = _poisson_pmf_matrix(_blend_with_baseline(players, "expected_goals_per_90") * fixture_mult, MAX_GOALS)
    assist_pmf = _poisson_pmf_matrix(_blend_with_baseline(players, "expected_assists_per_90") * fixture_mult, MAX_ASSISTS)
    clean_sheet_prob = np.clip(
        np.exp(-_blend_with_baseline(players, "expected_goals_conceded_per_90")) * fixture_mult, 0.0, 1.0
    )

    goals = np.arange(MAX_GOALS + 1)[:, None, None]
    assists = np.arange(MAX_ASSISTS + 1)[None, :, None]
    kept_clean_sheet = np.arange(2)[None, None, :]

    # (n_jugadores, goles, asistencias, clean_sheet)
    points = (
        APPEARANCE_POINTS
        + goals * goal_points[:, None, None, None]
        + assists * ASSIST_POINTS
        + kept_clean_sheet * clean_sheet_points[:, None, None, None]
    )
    cs_pmf = np.stack([1 - clean_sheet_prob, clean_sheet_prob], axis=1)
    probs = (
        goal_pmf[:, :, None, None]
        * assist_pmf[:, None, :, None]
        * cs_pmf[:, None, None, :]
    )

    n_players = len(players)
    points = points.reshape(n_players, -1)
    probs = probs.reshape(n_players, -1)

    # El jugador podría no jugar: ese escenario aporta 0 puntos y se lleva
    # la probabilidad restante.
    availability = _availability(players).to_numpy()
    probs = probs * availability[:, None]
    points = np.concatenate([np.zeros((n_players, 1)), points], axis=1)
    probs = np.concatenate([(1 - availability)[:, None], probs], axis=1)
    return points, probs


def _distribution_percentile(points: np.ndarray, probs: np.ndarray, percentile: float) -> np.ndarray:
    """Percentil `percentile` de cada fila de la distribución.

    Ordena los resultados por puntos, acumula probabilidad y devuelve el
    primer valor donde el acumulado alcanza el percentil pedido.
    """
    order = np.argsort(points, axis=1)
    sorted_points = np.take_along_axis(points, order, axis=1)
    cumulative = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    # `searchsorted` no admite un eje, así que se resuelve con una máscara:
    # el primer índice cuyo acumulado supera el umbral.
    reached = cumulative >= percentile
    idx = np.argmax(reached, axis=1)
    return sorted_points[np.arange(len(points)), idx]


def add_ceiling_metrics(players: pd.DataFrame) -> pd.DataFrame:
    """Agrega el techo (`xp_ceiling`) y la probabilidad de haul
    (`haul_probability`) — la varianza que el xP promedio no captura.

    - `xp_ceiling`: percentil `CEILING_PERCENTILE` de puntos. "En su 10%
      de mejores partidos, saca al menos esto."
    - `haul_probability`: P(puntos >= `HAUL_THRESHOLD`). Muy directo de
      comunicar y la métrica que de verdad decide una capitanía.
    """
    players = players.copy()
    points, probs = _points_distribution(players)
    players["xp_ceiling"] = _distribution_percentile(points, probs, CEILING_PERCENTILE).round(2)
    players["haul_probability"] = (probs * (points >= HAUL_THRESHOLD)).sum(axis=1).round(3)
    return players


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


def _load_last_season_baseline() -> pd.DataFrame | None:
    """Métricas de la temporada anterior, o None si no se guardaron.

    Existe porque FPL resetea todos los acumulados al arrancar una
    temporada: `starts`, `minutes` y las tasas por 90' vuelven a cero. Sin
    esta referencia el modelo se quedaría sin insumo — en GW2, el xG90 de
    un jugador saldría de un único partido, que es ruido, y un titular
    indiscutido sería indistinguible de un suplente.
    """
    if not LAST_SEASON_BASELINE_PATH.exists():
        return None
    return pd.read_csv(LAST_SEASON_BASELINE_PATH)


def effective_minutes(players: pd.DataFrame) -> pd.Series:
    """Minutos de evidencia disponible: los de esta temporada más los de
    la anterior (del baseline).

    `MIN_MINUTES_FOR_RANKING` filtra por muestra confiable, pero FPL
    resetea `minutes` a cero cada temporada. Usando solo los minutos
    actuales, en GW1 **nadie** pasa el filtro: verificado en la práctica
    — el pool elegible quedó en 0 jugadores y el ILP del Wildcard tiró
    `Infeasible`, que fue lo que rompió el pipeline.

    Un jugador con 3.000 minutos el año pasado es una muestra confiable
    aunque hoy lleve 90. La confiabilidad de la muestra no desaparece
    porque cambie el número de temporada.
    """
    current = players["minutes"].fillna(0)
    baseline = _load_last_season_baseline()
    if baseline is None:
        return current
    previous = players["id"].map(dict(zip(baseline["player_id"], baseline["minutes"]))).fillna(0)
    return current + previous


def _blend_with_baseline(players: pd.DataFrame, column: str) -> pd.Series:
    """Mezcla una tasa por 90' de la temporada en curso con la de la
    anterior, ponderando por minutos jugados.

    Al principio de la temporada manda la evidencia vieja; al llegar a
    `BASELINE_BLEND_MINUTES` pesan igual; después domina la nueva. Evita
    el salto brusco de "toda la información desaparece en GW1" sin
    quedarse anclado a datos viejos cuando ya hay muestra fresca.

    Sin baseline (o para un jugador que no figure en él — un debutante)
    devuelve la tasa actual tal cual.
    """
    current = players[column].fillna(0)
    baseline = _load_last_season_baseline()
    if baseline is None or column not in baseline.columns:
        return current

    previous = players["id"].map(dict(zip(baseline["player_id"], baseline[column])))
    minutes = players["minutes"].fillna(0)
    blended = (current * minutes + previous * BASELINE_BLEND_MINUTES) / (minutes + BASELINE_BLEND_MINUTES)
    return blended.fillna(current)


def save_last_season_baseline(players: pd.DataFrame) -> str:
    """Congela las métricas mientras todavía son de la temporada anterior
    — es decir, en pre-temporada.

    **Se escribe una sola vez**: si el archivo ya existe, no se toca. Es
    un snapshot único, y reescribirlo solo puede empeorarlo.

    El guard original miraba si algún partido figuraba como `finished`,
    y falló en la práctica: durante GW1, seis partidos ya se habían
    jugado (`started=True`) pero ninguno estaba marcado `finished`
    todavía — FPL lo marca recién cuando procesa los datos. En esa
    ventana los acumulados de los jugadores YA estaban reseteados, así
    que el guard dejó pasar la escritura y sobrescribió el baseline con
    ceros. Existir o no existir el archivo no depende del timing de FPL.
    """
    if LAST_SEASON_BASELINE_PATH.exists():
        return "El baseline ya existe — no se sobrescribe (es un snapshot de una sola vez)."

    baseline = pd.DataFrame({
        "player_id": players["id"],
        "web_name": players["web_name"],
        "start_rate": (players["starts"].fillna(0) / FULL_SEASON_MATCHES).clip(upper=1.0).round(3),
        "minutes": players["minutes"].fillna(0),
    })
    for column in BASELINE_RATE_COLUMNS:
        baseline[column] = players[column].fillna(0)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(LAST_SEASON_BASELINE_PATH, index=False)
    return f"Baseline de la temporada anterior guardado: {len(baseline)} jugadores, {len(BASELINE_RATE_COLUMNS) + 2} métricas."


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

    Suavizado (shrinkage) bayesiano: con pocos partidos jugados la tasa
    cruda es ruido — 2 de 2 no significa "100% titular".
    `START_RATE_PRIOR_WEIGHT` equivale a partidos de evidencia previa, así
    que la tasa arranca en el prior y se acerca a la real a medida que se
    acumulan partidos.

    El prior es **informado cuando se puede**: la tasa del jugador la
    temporada pasada (`save_last_season_baseline`), y solo cae a
    `START_RATE_PRIOR` para quien no tenga historial (un debutante, o un
    fichaje que llega de otra liga). Sin esto, el reseteo de `starts` al
    arrancar la temporada borraría de golpe la señal: en GW1 un titular
    indiscutido daría 0.79 y un suplente 0.62 — prácticamente lo mismo —
    y harían falta ~10 fechas para volver a distinguirlos.
    """
    played = get_matches_played_by_team()
    team_matches = players["team_name"].map(played).fillna(0)
    season_started = bool(team_matches.gt(0).any())
    if not season_started:
        # Pre-temporada: `starts` es de la temporada anterior, así que el
        # denominador correcto es esa temporada completa, no cero. Ya
        # empezada la temporada NO se fuerza: un equipo que todavía no
        # jugó debe quedar con denominador 0, lo que hace que la tasa
        # caiga limpiamente en su prior del baseline.
        team_matches = pd.Series(FULL_SEASON_MATCHES, index=players.index)

    if season_started:
        baseline = _load_last_season_baseline()
        if baseline is not None:
            # Solo se usa el baseline de quien acumuló minutos suficientes
            # la temporada pasada. Un `start_rate` de 0 puede significar
            # dos cosas muy distintas: "fue suplente" o "no jugó en esta
            # liga" — y tratar la segunda como la primera hunde a
            # fichajes, ascendidos y cedidos que hoy son titulares.
            # Detectado en datos reales: Rashford, Rudoni, Schlager y
            # otros arrancaron GW1/GW2 cargando un prior de 0.00 que
            # provenía de cero minutos registrados, no de suplencia.
            informative = baseline[baseline["minutes"] >= MIN_BASELINE_MINUTES]
            prior = players["id"].map(
                dict(zip(informative["player_id"], informative["start_rate"]))
            ).fillna(START_RATE_PRIOR)
        else:
            prior = START_RATE_PRIOR
    else:
        # En pre-temporada los propios `starts` ya son la evidencia del
        # año pasado; usar el baseline acá sería contar lo mismo dos veces.
        prior = START_RATE_PRIOR

    starts = players["starts"].fillna(0)
    return (
        (starts + START_RATE_PRIOR_WEIGHT * prior)
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
    fixtures["label"] = fixtures.apply(
        lambda r: f"{r['opponent']}({'H' if r['is_home'] else 'A'})", axis=1
    )
    summary = fixtures.sort_values("gameweek").groupby("team_name")["label"].agg(" · ".join)

    players["xp_horizon"] = gameweek_expected_points(players, horizon).sum(axis=1).round(2)
    players["fixtures_horizon"] = players["team_name"].map(summary).fillna("—")
    return players


def gameweek_expected_points(players: pd.DataFrame, horizon: int = HORIZON_GAMEWEEKS) -> pd.DataFrame:
    """El mismo xP del horizonte, pero **desglosado fecha por fecha** en vez
    de sumado. Devuelve un DataFrame con el índice de `players` y una
    columna por gameweek.

    `xp_horizon` es la suma de estas columnas — literalmente, para que no
    puedan divergir. Existe por separado porque hay decisiones donde el
    total no alcanza: en una posición solo juega un número fijo de
    titulares por fecha, así que para saber cuánto rinde REALMENTE un
    plantel hay que resolver semana a semana quién de los que tenés
    arranca. Sumar `xp_horizon` cuenta puntos de suplentes que nunca se
    van a cobrar (el caso extremo es el arquero: tenés dos, juega uno).

    Un blank gameweek queda en 0 y un double suma sus dos partidos, sin
    lógica especial: el multiplicador de la fecha es la suma de los
    partidos que el equipo juegue en ella.
    """
    fixtures = get_team_fixtures_horizon(horizon)
    fixtures["multiplier"] = _fixture_multiplier(fixtures["difficulty"])
    first_gw = fixtures["gameweek"].min()

    base_rate = _base_scoring_rate(players)
    start_rate = _start_rate(players)
    prob_next = start_rate * _playing_probability(players)
    later_availability = start_rate * (
        players["chance_of_playing_next_round"].fillna(0).div(100)
        .where(players["status"] != "a", 1.0)
    )

    by_gameweek = {}
    for gameweek, week in fixtures.groupby("gameweek"):
        multiplier = players["team_name"].map(week.groupby("team_name")["multiplier"].sum()).fillna(0.0)
        # La duda por lesión solo se toma literal para la próxima fecha;
        # de ahí en adelante se asume que un jugador sano juega normal y
        # que la duda de un lesionado persiste (ver docstring de arriba).
        availability = prob_next if gameweek == first_gw else later_availability
        by_gameweek[int(gameweek)] = base_rate * multiplier * availability
    return pd.DataFrame(by_gameweek, index=players.index)


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
        & (effective_minutes(players) >= MIN_MINUTES_FOR_RANKING)
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

    Muestra siempre `xp_ceiling` y `haul_probability` junto al xP medio,
    porque para capitanía el promedio engaña: verificado con datos
    reales, Raya (GKP) proyectaba MÁS xP que Haaland pero con 0% de
    probabilidad de haul — duplicar a un arquero no puede dar un
    resultado grande. Con `stance="chase"` el ranking pasa a ordenarse
    por techo en vez de por media: remontar posiciones necesita
    resultados grandes, no consistencia.

    Solo la capitanía usa el techo. Para transferencias sigue mandando el
    promedio a 4 fechas: ahí la varianza semana a semana se diluye, y lo
    que importa es el acumulado.
    """
    likely_to_play = players[
        (players["chance_of_playing_next_round"].fillna(100) >= 75)
        & (effective_minutes(players) >= MIN_MINUTES_FOR_RANKING)
    ]
    likely_to_play = add_rank_adjusted_value(likely_to_play, stance)
    if "xp_ceiling" not in likely_to_play.columns:
        likely_to_play = add_ceiling_metrics(likely_to_play)

    sort_column = "xp_ceiling" if stance == "chase" else "rank_value"
    columns = [
        "web_name", "team_name", "position", "now_cost", "xp_next", "xp_ceiling", "haul_probability",
        "selected_by_percent", "rank_value", "next_opponent", "next_is_home",
    ]
    return likely_to_play.sort_values(sort_column, ascending=False).head(top_n)[columns]


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
    players = add_ceiling_metrics(add_expected_points(players))
    save_scored_players(players)

    print("\n=== Top 5 opciones de capitanía — próximo gameweek ===")
    print(captaincy_picks(players).to_string(index=False))

    print("\n=== Top 10 diferenciales (ownership <= 10%) ===")
    print(top_differentials(players).to_string(index=False))


if __name__ == "__main__":
    run_analysis()
