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

- **base_rate**: puntos esperados por 90 minutos, calculados desde
  estadísticas subyacentes — `expected_goals_per_90 × puntos_por_gol(posición)`
  `+ expected_assists_per_90 × 3` (ataque), más
  `exp(-expected_goals_conceded_per_90) × puntos_por_clean_sheet(posición)`
  (defensa, con la probabilidad de clean sheet estimada vía Poisson), más
  2 puntos fijos de aparición. **No usa `form`/`points_per_game`** —
  puntos ya anotados mezclan la calidad real del jugador con suerte
  puntual (un bonus de un partido aislado no dice nada sobre si se
  repite); xG/xA reflejan las oportunidades generadas, un proxy más
  estable. Limitación conocida: no modela bonus points (BPS) ni atajadas
  de arquero — FPL no expone un "bono esperado" por jugador.
- **fixture_multiplier**: invierte el FDR de FPL (1=fácil, 5=difícil),
  centrado en 1.0 para dificultad media (3).
- **playing_probability**: `chance_of_playing_next_round` / 100, o 100% si
  no hay duda de lesión.

Jugadores con menos de `MIN_MINUTES_FOR_RANKING` (900 min, ~10 partidos) se
excluyen de diferenciales y capitanía — con poca muestra, el promedio de
puntos es ruido, no señal.

## Cómo funciona la recomendación de equipo

Armar los 15 titulares de un plantel (`select_squad`) es un problema de
optimización con restricciones cruzadas (presupuesto, exactamente
2 GKP/5 DEF/5 MID/3 FWD, máx. 3 por club, sin enfrentamientos internos —
ver abajo). Se resuelve con programación lineal entera (`PuLP`), que
garantiza el óptimo real — no una aproximación. El Wildcard usa esto
directo (recalculado desde cero cada vez); el Base lo usa solo como punto
de partida — ver la sección siguiente, es la pieza más importante del
proyecto.

**11 titulares** (`select_starting_xi`): de los 15 ya elegidos (Base o
Wildcard), se prueban las ~13 formaciones válidas (3-5 DEF, 2-5 MID,
1-3 FWD) y se toma la de mayor xP. Al ser un espacio de búsqueda chico,
alcanza con fuerza bruta — no hace falta un solver acá.

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

**Simplificación conocida:** cuando se arma desde cero (Wildcard, o el
Base la primera vez), el modelo maximiza el xP total de los 15, no
específicamente el de los 11 mejores. En la práctica esto tiende a
"gastar de más" en el 5to defensor o 2do arquero en vez de priorizar
jugadores baratos y confiables para el banco.

## Equipo Base vs Equipo Wildcard

Dos filosofías completamente distintas, no solo dos parámetros distintos
del mismo cálculo:

- **Wildcard** (`select_wildcard_squad`): un ejercicio *sin memoria* —
  "si pudiera rearmar todo desde cero hoy, con las reglas de FPL, ¿cuál es
  el mejor equipo posible?". Se recalcula 100% desde cero cada vez, sin
  importar qué recomendó ayer. Como el banco no necesita ser jugable
  (nunca se sostiene más de una fecha), gasta lo mínimo posible ahí y
  vuelca el resto del presupuesto a los 11 titulares. Un solo ILP decide
  plantel y titulares a la vez (variables `in_squad` e `in_xi`, con
  `in_xi <= in_squad`): maximizar el xP del XI es la prioridad absoluta
  (peso `WILDCARD_XI_WEIGHT`), y minimizar el costo total del plantel es
  el criterio de desempate. Los enfrentamientos internos solo se evalúan
  entre titulares — un suplente que no juega no le puede romper el clean
  sheet a nadie.

- **Base** (`evolve_base_squad` en `transfer_advisor.py`): el equipo que
  **de verdad vas a usar en FPL**, con memoria real. No se recalcula desde
  cero — parte del equipo de la fecha anterior (persistido en
  `my_team.csv`) y el modelo decide, con la misma lógica del asesor de
  transferencias (ver abajo), si vale la pena mover algo esta semana. Si
  no hay una mejora que justifique gastar la transferencia, el Base se
  queda exactamente igual. Si la hay, aplica el cambio — hits de -4
  incluidos, si el modelo los considera rentables (ver "Asesor de
  transferencias"). Corre una sola vez por gameweek (no una vez por
  corrida diaria): un archivo de estado
  (`data/processed/base_squad_state.json`) guarda la última fecha
  evaluada y cuántas transferencias tiene acumuladas, para no "gastar"
  cambios distintos cada día dentro de la misma semana — algo que en el
  juego real no existe.

En la práctica, el XI del Wildcard suele sacar más xP que el del Base
(mismo presupuesto, banco mucho más barato) — es la comparación esperada:
uno es el techo teórico sin restricciones reales, el otro es el equipo
real, restringido por lo que ya comprás y cuántas transferencias tenés.

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

Dos formas de usar la misma lógica de decisión:

1. **Automática** (`evolve_base_squad`, corre dentro de `run_squad.py` y
   `export_for_tableau.py`): aplica los cambios directo sobre
   `my_team.csv`, sin pedir confirmación. Es lo que mantiene al equipo
   Base al día — ver la sección anterior.
2. **Manual/interactiva** (`python scripts/run_transfer_advisor.py --team
   my_team.csv --bank 0.5 --free-transfers 1`): la misma lógica, pero solo
   imprime la recomendación para que la leas — no toca `my_team.csv`. Útil
   para simular "¿qué pasaría si tuviera este banco/estas transferencias?"
   sin afectar el estado real del Base.

Ambas responden las mismas preguntas:

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
- **Si el que entra terminaría en el banco**, su xP no se cobra mientras
  esté ahí — la ganancia proyectada se descuenta (`BENCH_GAIN_DISCOUNT =
  0.3`) en vez de contarse entera. El cambio todavía es posible si la
  ventaja alcanza para justificarlo igual, pero el umbral efectivo sube —
  así que en la práctica, casi siempre es mejor guardar la transferencia
  para un movimiento que sí sea titular. (Se agregó después de detectar
  en la práctica que el primer cambio real dejó al jugador nuevo en el
  banco, gastando la transferencia sin que sus puntos contaran para nada.)

El xP a horizonte suma los partidos reales de cada equipo por fecha, así que
double gameweeks (dos partidos) y blanks (ninguno) quedan contados
naturalmente. La duda de disponibilidad de un lesionado se asume persistente
en el horizonte — conservador a propósito.

El timing de chips no se optimiza (requeriría proyectar la temporada
completa): hay reglas guía en `docs/chips-strategy.md`, y el asesor avisa
cuando detecta un double gameweek en el horizonte.

Filosofía: **prepararse, no predecir** — el equipo Base itera fecha a fecha
con datos frescos; no arma un plan de 10 fechas que la realidad va a romper
en la primera. `my_team.csv` es público en el repo a propósito, junto con
todos los demás CSVs — el Project de claude.ai también puede analizarlo.

## Vista especulativa a 4 fechas

`python scripts/run_trajectory_preview.py` genera
`data/processed/squad_trajectory_preview.csv`: 15 filas (un "slot" del
plantel actual) × columnas `GW1`...`GW4` con el nombre del jugador en ese
slot cada fecha. Si un slot no cambia, el nombre se repite en las columnas
siguientes; si cambia, se ve el nombre nuevo desde esa columna en
adelante — fácil de leer de un vistazo.

**Importante — esto NO es una predicción**, es una simulación: corre la
misma lógica de `evolve_base_squad` (banco-vs-usar, hit-vs-no-hit) cuatro
veces seguidas usando el `xp_horizon` de **hoy**, sin esperar datos nuevos
entre fecha y fecha simulada. En la vida real, cada gameweek trae datos
frescos (lesiones, precios, forma) que probablemente cambien la decisión
real cuando de verdad llegue esa fecha — por eso es "el plan de hoy, si
nada cambiara", no un compromiso. Es de solo lectura: nunca toca
`my_team.csv` ni `base_squad_state.json`, son completamente independientes.

## Exports para Tableau

`python scripts/export_for_tableau.py` genera dos CSVs en `data/processed/`:

- **`players_scored.csv`**: los ~590 jugadores con xP, precio, ownership,
  próximo rival, etc. — para dashboards sobre todo el pool.
- **`squad_recommendations.csv`**: Base + Wildcard combinados en formato
  tidy (una fila por jugador, columnas `squad_type`/`role`/`is_captain`
  para filtrar) — una sola fuente de datos para comparar ambos equipos.

Ambos incluyen `xp_horizon` y `fixtures_horizon` (el xP y el calendario
resumido de las próximas 4 fechas, no solo la siguiente — pensado para que
se pueda explicar el *por qué* de una elección, no solo el qué: por
ejemplo, por qué el asesor prefirió a un jugador sobre otro que tenía mejor
xP inmediato pero un calendario peor a mediano plazo).

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
