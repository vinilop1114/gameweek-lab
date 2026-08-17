import json

import pandas as pd

from gameweek_lab.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, PLAYER_PHOTO_URL_TEMPLATE


def _load_raw(name: str):
    path = DATA_RAW_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _next_gameweek(fixtures: list[dict]) -> int:
    upcoming_events = sorted({f["event"] for f in fixtures if not f["finished"] and f["event"] is not None})
    return upcoming_events[0]


def _next_fixture_by_team(fixtures: list[dict]) -> dict[int, dict]:
    """Para cada equipo, su próximo fixture sin jugar (el de gameweek más bajo)."""
    upcoming = [f for f in fixtures if not f["finished"] and f["event"] is not None]
    upcoming.sort(key=lambda f: f["event"])

    next_fixture: dict[int, dict] = {}
    for fixture in upcoming:
        for team_id, is_home in ((fixture["team_h"], True), (fixture["team_a"], False)):
            if team_id in next_fixture:
                continue  # ya guardamos el próximo fixture de este equipo, era el más temprano
            opponent_id = fixture["team_a"] if is_home else fixture["team_h"]
            difficulty = fixture["team_h_difficulty"] if is_home else fixture["team_a_difficulty"]
            next_fixture[team_id] = {
                "gameweek": fixture["event"],
                "opponent_id": opponent_id,
                "is_home": is_home,
                "difficulty": difficulty,
            }
    return next_fixture


def get_next_gameweek_fixtures() -> pd.DataFrame:
    """Los partidos del próximo gameweek: quién juega contra quién.

    La usa squad_builder para evitar que el plantel tenga, a la vez, al
    defensor/arquero de un equipo y a un mediocampista/delantero del rival
    que enfrenta ese mismo gameweek — un jugador propio le rompería el
    clean sheet al otro.
    """
    bootstrap = _load_raw("bootstrap-static")
    fixtures = _load_raw("fixtures")
    team_names = pd.DataFrame(bootstrap["teams"]).set_index("id")["name"]

    next_gw = _next_gameweek(fixtures)
    rows = [
        {
            "gameweek": f["event"],
            "team_h_name": team_names[f["team_h"]],
            "team_a_name": team_names[f["team_a"]],
        }
        for f in fixtures
        if f["event"] == next_gw
    ]
    return pd.DataFrame(rows)


def build_players_dataset() -> pd.DataFrame:
    bootstrap = _load_raw("bootstrap-static")
    fixtures = _load_raw("fixtures")

    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"]).set_index("id")
    positions = pd.DataFrame(bootstrap["element_types"]).set_index("id")

    # La API manda varios números como texto (ej. "4.4") — hay que convertirlos
    numeric_as_text = ["form", "points_per_game", "selected_by_percent", "ep_next", "ep_this"]
    for col in numeric_as_text:
        players[col] = pd.to_numeric(players[col], errors="coerce")
    players["now_cost"] = players["now_cost"] / 10.0  # el precio viene x10 (60 = £6.0m)

    players["team_name"] = players["team"].map(teams["name"])
    players["position"] = players["element_type"].map(positions["singular_name_short"])
    players["full_name"] = players["first_name"] + " " + players["second_name"]
    players["photo_url"] = players["code"].map(lambda code: PLAYER_PHOTO_URL_TEMPLATE.format(code=code))

    next_fixture = _next_fixture_by_team(fixtures)
    players["next_gameweek"] = players["team"].map(lambda t: next_fixture.get(t, {}).get("gameweek"))
    players["next_opponent"] = players["team"].map(
        lambda t: teams["name"].get(next_fixture.get(t, {}).get("opponent_id"))
    )
    players["next_is_home"] = players["team"].map(lambda t: next_fixture.get(t, {}).get("is_home"))
    players["next_fixture_difficulty"] = players["team"].map(lambda t: next_fixture.get(t, {}).get("difficulty"))

    columns = [
        "id", "web_name", "full_name", "team_name", "position", "now_cost", "status", "photo_url",
        "total_points", "points_per_game", "form", "selected_by_percent",
        "minutes", "chance_of_playing_next_round", "ep_next",
        "next_gameweek", "next_opponent", "next_is_home", "next_fixture_difficulty",
    ]
    players = players[columns].sort_values("total_points", ascending=False).reset_index(drop=True)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "players.csv"
    players.to_csv(out_path, index=False)
    print(f"Guardado {out_path} ({len(players)} jugadores)")
    return players


if __name__ == "__main__":
    build_players_dataset()
