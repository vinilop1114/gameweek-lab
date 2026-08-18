import pandas as pd

from gameweek_lab.build_dataset import build_players_dataset, get_team_fixtures_horizon
from gameweek_lab.config import DATA_PROCESSED_DIR

# Debajo de esto, 'points_per_game' es ruido: pocos partidos alcanzan para
# que una sola actuación puntuda dispare el promedio sin que sea repetible.
# 900 minutos ~ 10 partidos completos.
MIN_MINUTES_FOR_RANKING = 900

# Cuántas fechas adelante mira el asesor de transferencias. Más allá de
# ~4 las proyecciones de nuestra heurística pierden sentido — la idea es
# prepararse para las próximas fechas e iterar, no predecir la temporada.
HORIZON_GAMEWEEKS = 4


def _base_scoring_rate(players: pd.DataFrame) -> pd.Series:
    """Puntos esperados 'crudos' por partido, antes de ajustar por rival.

    Usamos 'form' (promedio de los últimos partidos) cuando hay datos de la
    temporada en curso. Pre-temporada, form es 0.0 para todos los jugadores
    (todavía no se jugó nada), así que ahí caemos a 'points_per_game' de la
    temporada pasada como mejor proxy disponible.
    """
    has_current_form = players["form"] > 0
    return players["form"].where(has_current_form, players["points_per_game"])


def _fixture_multiplier(difficulty: pd.Series) -> pd.Series:
    """Convierte el FDR de FPL (1=fácil, 5=difícil) en un multiplicador.

    Centrado en dificultad 3 (multiplicador 1.0): rivales más fáciles suben
    el xP, más difíciles lo bajan. Rango aprox. 0.33x (rival top) a 1.67x
    (rival más débil).
    """
    return (6 - difficulty) / 3


def _playing_probability(players: pd.DataFrame) -> pd.Series:
    """chance_of_playing_next_round viene en % (0-100) o NaN si no hay duda
    de lesión/rotación — en ese caso asumimos 100% de probabilidad de jugar.
    """
    return players["chance_of_playing_next_round"].fillna(100) / 100


def add_expected_points(players: pd.DataFrame) -> pd.DataFrame:
    players = players.copy()
    base_rate = _base_scoring_rate(players)
    fixture_mult = _fixture_multiplier(players["next_fixture_difficulty"])
    playing_prob = _playing_probability(players)

    players["xp_next"] = (base_rate * fixture_mult * playing_prob).round(2)
    return players


def add_horizon_expected_points(players: pd.DataFrame, horizon: int = HORIZON_GAMEWEEKS) -> pd.DataFrame:
    """Agrega xp_horizon: puntos esperados acumulados en las próximas
    `horizon` fechas, sumando el multiplicador de dificultad de cada
    partido del equipo (double gameweeks suman dos partidos, blanks cero).

    Disponibilidad: para la primera fecha usamos chance_of_playing_next_round
    como en xp_next. Para las siguientes asumimos que un jugador sano ('a')
    juega normal, y que la duda de un lesionado/suspendido persiste — es
    conservador, pero un asesor de transferencias DEBE penalizar tener
    jugadores que quizás no jueguen.
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
    prob_next = _playing_probability(players)
    later_availability = (
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


def captaincy_picks(players: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Mejores opciones de capitán para el próximo gameweek: alto xP, buena
    probabilidad de jugar, y minutos suficientes para confiar en el dato."""
    likely_to_play = players[
        (players["chance_of_playing_next_round"].fillna(100) >= 75)
        & (players["minutes"] >= MIN_MINUTES_FOR_RANKING)
    ]
    columns = ["web_name", "team_name", "now_cost", "xp_next", "next_opponent", "next_is_home"]
    return likely_to_play.sort_values("xp_next", ascending=False).head(top_n)[columns]


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
