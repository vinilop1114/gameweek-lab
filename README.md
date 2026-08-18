# The Gameweek Lab — FPL Data Pipeline

Scripts locales en Python para analizar datos de Fantasy Premier League (FPL),
como base de contenido para la cuenta de Instagram "The Gameweek Lab".

## Uso

```bash
pip install -r requirements.txt

python scripts/run_fetch.py           # descarga datos frescos de la API de FPL
python scripts/run_analysis.py        # calcula xP, diferenciales y capitanía
python scripts/run_squad.py           # equipo Base: sostenible para varias fechas
python scripts/run_wildcard_squad.py  # equipo Wildcard: banco mínimo, XI máximo
python scripts/export_for_tableau.py  # CSVs listos para Tableau (ver abajo)
python scripts/run_transfer_advisor.py --team my_team.csv --bank 0.5 --free-transfers 1
```

## Estructura

```
gameweek_lab/
  config.py          # rutas y URLs de la API
  fetch.py            # descarga bootstrap-static y fixtures → data/raw/*.json
  build_dataset.py    # limpia y combina el JSON crudo → data/processed/players.csv
  analysis.py          # expected points (xP), diferenciales, capitanía
  squad_builder.py     # plantel de 15 (PuLP/ILP) + 11 titulares (fuerza bruta sobre formaciones)
data/
  raw/                # snapshots JSON tal cual los devuelve la API
  processed/           # players.csv, listo para analizar
scripts/
  run_fetch.py         # entry point: descarga
  run_analysis.py       # entry point: análisis
```

## Cómo funciona el cálculo de xP

`xp_next = base_rate × fixture_multiplier × playing_probability`

- **base_rate**: `form` (promedio reciente) si ya hay partidos jugados esta
  temporada; si no (ej. pre-temporada, `form` en 0 para todos), usa
  `points_per_game` de la temporada anterior como proxy.
- **fixture_multiplier**: invierte el FDR de FPL (1=fácil, 5=difícil),
  centrado en 1.0 para dificultad media (3).
- **playing_probability**: `chance_of_playing_next_round` / 100, o 100% si
  no hay duda de lesión.

Jugadores con menos de `MIN_MINUTES_FOR_RANKING` (900 min, ~10 partidos) se
excluyen de diferenciales y capitanía — con poca muestra, el promedio de
puntos es ruido, no señal.

**Nota:** este xP es una heurística simple para aprender el proceso, no un
modelo predictivo serio. FPL ya expone su propia estimación en el campo
`ep_next` de la API — vale la pena comparar contra eso más adelante.

## Cómo funciona la recomendación de equipo

Dos pasos, dos técnicas distintas:

1. **Plantel de 15** (`select_squad`): problema de optimización con
   restricciones cruzadas (presupuesto, exactamente 2 GKP/5 DEF/5 MID/3 FWD,
   máx. 3 por club, sin enfrentamientos internos — ver abajo). Se resuelve
   con programación lineal entera (`PuLP`), que garantiza el óptimo real —
   no una aproximación.
2. **11 titulares** (`select_starting_xi`): de los 15 ya elegidos, se prueban
   las ~13 formaciones válidas (3-5 DEF, 2-5 MID, 1-3 FWD) y se toma la de
   mayor xP. Al ser un espacio de búsqueda chico, alcanza con fuerza bruta —
   no hace falta un solver acá.

Solo se consideran jugadores con `status == "a"` (disponibles, sin lesión ni
suspensión) y al menos `MIN_MINUTES_FOR_RANKING` minutos — mismo filtro que
en diferenciales/capitanía, por la misma razón: evitar que un jugador con
muestra chica (ej. un arquero con un solo partido bueno) desplace del
plantel a un titular real por ruido estadístico.

**Enfrentamientos internos — permitidos, pero pagan su precio:** tener al
defensor/arquero de un equipo y a un mediocampista/delantero del rival que
enfrenta ese mismo gameweek tiene un costo real (si el atacante anota, mata
el clean sheet del propio defensor). En vez de prohibir esas duplas, el
modelo les descuenta ese costo esperado (`CLASH_PENALTY_XP` ≈ 0.5 xP,
ver `_internal_clash_penalty` en `squad_builder.py`): una dupla que proyecte
ganar más que su penalización entra al equipo igual. El auto-sabotaje está
permitido cuando los números lo justifican. En el Wildcard la penalización
solo mira a los titulares — un suplente que no juega no le rompe el clean
sheet a nadie.

**Filosofía:** este plantel es la base de la temporada, no una apuesta a un
solo gameweek — por eso importa que no se auto-sabotee. Los ajustes semana
a semana (fixtures que cambian, forma que sube o baja) se resuelven con las
transferencias normales de FPL, no recalculando todo el equipo de cero.

**Simplificación conocida:** el equipo Base maximiza el xP total de los 15,
no específicamente el de los 11 mejores. En la práctica esto tiende a
"gastar de más" en el 5to defensor o 2do arquero en vez de priorizar
jugadores baratos y confiables para el banco.

## Equipo Base vs Equipo Wildcard

Dos recomendaciones con objetivos distintos, para dos audiencias distintas:

- **Base** (`select_squad` + `select_starting_xi`, ya descrito arriba):
  reparte el presupuesto entre los 15, pensado para sostenerse varias
  fechas con las transferencias normales de FPL.
- **Wildcard** (`select_wildcard_squad`): asume que se puede rearmar el
  plantel entero cada fecha, así que el banco no necesita ser jugable —
  gasta lo mínimo posible ahí y vuelca el resto del presupuesto a los 11
  titulares. Un solo ILP decide plantel y titulares a la vez (variables
  `in_squad` e `in_xi`, con `in_xi <= in_squad`): maximizar el xP del XI
  es la prioridad absoluta (peso `WILDCARD_XI_WEIGHT`), y minimizar el
  costo total del plantel es el criterio de desempate que empuja el gasto
  del banco al mínimo. Los enfrentamientos internos solo se evalúan entre
  titulares — un suplente que no juega no le puede romper el clean sheet
  a nadie.

En la práctica, el XI del Wildcard saca más xP que el del Base (mismo
presupuesto, mejor aprovechado) a costa de un banco que, si tuvieras que
usarlo, rendiría mucho peor.

**Capitán, vice y orden del banco:** cada equipo recomienda capitán (mayor
xP del XI) y vice-capitán (segundo mayor — hereda la cinta si el capitán no
juega). El banco sale ordenado según la mecánica real de FPL: el arquero
suplente en su slot fijo (los arqueros solo se intercambian entre sí) y los
tres de campo por xP descendente — poner al mejor primero no arriesga nada,
porque cuando un titular no juega FPL recorre el banco en orden y saltea
automáticamente a quien rompería la formación mínima (≥3 DEF, ≥2 MID,
≥1 FWD). La salida incluye una simulación de auto-suplencias: para cada
titular, quién entraría si no juega, ya validado contra esa regla.

## Asesor de transferencias

`python scripts/run_transfer_advisor.py --team my_team.csv --bank 0.5 --free-transfers 1`

Le das tu equipo actual (copiá `my_team.example.csv` a `my_team.csv` y poné
tus 15 jugadores por `web_name`, con `team_name` para desambiguar), y responde:

- **El mejor cambio disponible** esta fecha, rankeado por ganancia de xP en
  las próximas `HORIZON_GAMEWEEKS` fechas (4 por defecto) — no solo la
  próxima, así el cambio "se prepara" para el calendario que viene.
- **Si conviene hacerlo o guardar la transferencia**: mejoras marginales
  (< `BANK_THRESHOLD` xP en el horizonte) no justifican gastar la
  transferencia — acumularla (hasta 5) habilita movimientos dobles después.
- **Si un hit de -4 se justifica**: solo si el segundo mejor cambio gana más
  de 4 + `HIT_UNCERTAINTY_MARGIN` puntos proyectados — pagar puntos ciertos
  por una proyección requiere margen de seguridad, no empate técnico.
- **Jugadores con bandera** (lesión/suspensión/duda) se marcan siempre, y
  un cambio que saca a uno de ellos se recomienda aunque la ganancia sea chica.

El xP a horizonte suma los partidos reales de cada equipo por fecha, así que
double gameweeks (dos partidos) y blanks (ninguno) quedan contados
naturalmente. La duda de disponibilidad de un lesionado se asume persistente
en el horizonte — conservador a propósito.

El timing de chips no se optimiza (requeriría proyectar la temporada
completa): hay reglas guía en `docs/chips-strategy.md`, y el asesor avisa
cuando detecta un double gameweek en el horizonte.

Filosofía: **prepararse, no predecir** — el asesor se corre cada semana con
datos frescos e itera; no arma un plan de 10 fechas que la realidad va a
romper en la primera.

## Exports para Tableau

`python scripts/export_for_tableau.py` genera dos CSVs en `data/processed/`:

- **`players_scored.csv`**: los 587 jugadores con xP, precio, ownership,
  próximo rival, etc. — para dashboards sobre todo el pool.
- **`squad_recommendations.csv`**: Base + Wildcard combinados en formato
  tidy (una fila por jugador, columnas `squad_type`/`role`/`is_captain`
  para filtrar) — una sola fuente de datos para comparar ambos equipos.

Ambos incluyen `photo_url` (foto del jugador desde el CDN oficial de
Premier League, armada con el campo `code` de la API — ver
`PLAYER_PHOTO_URL_TEMPLATE` en `config.py`). Para que Tableau la muestre
como imagen: clic derecho sobre `photo_url` en el panel de datos →
Default Properties → Image Role.

**Tamaño de imagen:** usamos 40x40 (no 110x140 ni 250x250) porque Tableau
Public rechaza imágenes de más de 128KB, y los tamaños más grandes lo
superan en algunos casos (depende de cuánto detalle tenga la foto
puntual, no solo de la resolución). 40x40 tiene margen de sobra siempre.

**Jugadores sin foto:** ~35% de los jugadores no tienen foto todavía en
el CDN de la Premier League — sobre todo fichajes grandes de este
mercado (Wirtz, Donnarumma, Cherki, Zubimendi...) sin sesión oficial
todavía. `export_for_tableau.py` chequea cada URL contra la API en
paralelo (`gameweek_lab/photos.py`, `resolve_photo_urls`) y reemplaza
las que fallan por la silueta genérica oficial de FPL
(`Photo-Missing.png`, la misma que usa fantasy.premierleague.com) — así
ninguna celda queda con el ícono roto. Este chequeo solo corre en el
export para Tableau (~7s con ~590 jugadores, usando una sesión HTTP con
pool de conexiones); el resto del pipeline no depende de red extra.
