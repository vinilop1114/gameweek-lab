# The Gameweek Lab — Documentación Técnica

Pipeline local en Python que descarga datos oficiales de Fantasy Premier
League (FPL), calcula puntos esperados (xP) por jugador, arma recomendaciones
de equipo, y exporta todo en CSVs listos para visualizar en Tableau. Sirve
de base de contenido para la cuenta de Instagram "The Gameweek Lab".

Este documento describe **qué se construyó y por qué**, para acompañar al
código como referencia técnica. Para instrucciones rápidas de uso, ver
[README.md](README.md).

## Stack tecnológico

| Tecnología | Rol en el proyecto |
|---|---|
| **Python 3.14** | Lenguaje del proyecto |
| **pandas** | Estructura central de datos (`DataFrame`) para limpiar, cruzar y transformar la información de jugadores/fixtures |
| **requests** | Cliente HTTP — consume la API de FPL y verifica URLs de fotos |
| **PuLP** | Modelado de programación lineal entera (ILP); expresa las reglas de armado de equipo (presupuesto, posiciones, límite de club) como restricciones matemáticas |
| **CBC (COIN-OR Branch and Cut)** | Solver que PuLP invoca por debajo (`PULP_CBC_CMD`) para resolver los problemas de optimización — viene incluido con PuLP, no se instala aparte |
| **concurrent.futures.ThreadPoolExecutor** (librería estándar) | Paraleliza las ~590 verificaciones de fotos de jugadores |
| **json, pathlib, datetime** (librería estándar) | Lectura/escritura de datos crudos, manejo de rutas independiente del directorio de ejecución, timestamps |
| **API pública de Fantasy Premier League** | Fuente de datos: `bootstrap-static/` (jugadores, equipos, posiciones) y `fixtures/` (calendario, dificultad de partidos) |
| **CDN de fotos de la Premier League** (`resources.premierleague.com`) | Fuente de las fotos de jugadores usadas en las visualizaciones |
| **CSV** | Formato de intercambio entre este pipeline y Tableau |
| **Tableau Public** | Consumidor final de los datos — no forma parte del código, pero condicionó decisiones de diseño (ver sección de fotos) |

No hay control de versiones (Git) configurado todavía en este directorio.

## Estructura del proyecto

```
FPL/
├── gameweek_lab/              # paquete Python con toda la lógica
│   ├── config.py               # rutas, URLs, constantes de negocio
│   ├── fetch.py                 # descarga datos crudos de la API → JSON
│   ├── build_dataset.py        # JSON crudo → DataFrame limpio de jugadores
│   ├── analysis.py              # cálculo de xP, diferenciales, capitanía
│   ├── squad_builder.py         # optimización de plantel (ILP) — Base y Wildcard
│   └── photos.py                 # verificación/fallback de fotos de jugadores
├── data/
│   ├── raw/                    # snapshots JSON tal cual los devuelve la API
│   └── processed/               # CSVs limpios y calculados
├── scripts/                     # puntos de entrada ejecutables desde terminal
│   ├── run_fetch.py
│   ├── run_analysis.py
│   ├── run_squad.py
│   ├── run_wildcard_squad.py
│   └── export_for_tableau.py
└── requirements.txt
```

**Por qué esta separación:** `gameweek_lab/` contiene funciones puras y
reutilizables; `scripts/` son los comandos que efectivamente se corren.
Separar "lógica" de "cómo se invoca" permite reusar las mismas funciones
desde distintos entry points (por ejemplo, `export_for_tableau.py` reusa
`build_players_dataset`, `add_expected_points`, `select_squad`, etc. sin
duplicar código).

## Flujo de datos

```
API de FPL (bootstrap-static, fixtures)
        │  fetch.py
        ▼
  data/raw/*.json  (snapshot crudo)
        │  build_dataset.py
        ▼
  DataFrame de jugadores limpio (precios, posiciones, próximo fixture, foto)
        │  analysis.py (add_expected_points)
        ▼
  + columna xp_next
        │
        ├─→ analysis.py: top diferenciales, mejores capitanes
        │
        └─→ squad_builder.py (PuLP/ILP)
                ├─→ Equipo Base (sostenible, 15 jugadores)
                └─→ Equipo Wildcard (banco mínimo, XI máximo)
                        │  photos.py (resolve_photo_urls)
                        ▼
              data/processed/*.csv  →  Tableau
```

## Módulos, en detalle

### `config.py`
Centraliza rutas y URLs. `PROJECT_ROOT` se calcula desde la ubicación del
propio archivo (`Path(__file__).resolve().parent.parent`), así los scripts
funcionan sin importar desde qué carpeta se ejecuten.

### `fetch.py`
Descarga `bootstrap-static/` (jugadores, equipos, posiciones) y
`fixtures/` (calendario completo con dificultad por partido) y los guarda
crudos en `data/raw/`. Usa `response.raise_for_status()` para fallar
ruidosamente ante errores HTTP en vez de continuar con datos vacíos, y
`timeout=15` para no colgarse si la API no responde.

### `build_dataset.py`
Transforma el JSON crudo en un `DataFrame` de jugadores limpio:
- Convierte campos que la API manda como texto (`"4.4"`) a numéricos.
- Corrige el precio (`now_cost` viene multiplicado por 10).
- Cruza cada jugador con el nombre de su equipo, su posición, y su
  **próximo fixture** (rival, si juega de local, y la dificultad FDR que
  ya calcula FPL en escala 1-5).
- Arma `photo_url` a partir del campo `code` de cada jugador (ver sección
  de fotos más abajo).

También expone `get_next_gameweek_fixtures()`: la lista de partidos del
próximo gameweek (quién juega contra quién), usada por `squad_builder.py`
para detectar enfrentamientos internos.

### `analysis.py`
Calcula `xp_next` (puntos esperados para el próximo gameweek) con una
heurística simple y explicable:

```
xp_next = base_rate × fixture_multiplier × playing_probability
```

- **base_rate**: `form` (forma reciente) si ya se jugó esta temporada; si
  no (pre-temporada, `form = 0` para todos), usa `points_per_game` de la
  temporada anterior como proxy.
- **fixture_multiplier**: invierte el FDR de FPL, centrado en 1.0 para
  dificultad media.
- **playing_probability**: probabilidad de jugar según
  `chance_of_playing_next_round` (100% si no hay duda física).

`MIN_MINUTES_FOR_RANKING = 900` (~10 partidos): jugadores por debajo de
este umbral se excluyen de diferenciales, capitanía y armado de equipo.
Se descubrió empíricamente que sin este filtro, jugadores con un solo
partido bueno (ej. un arquero con 90 minutos jugados) dominaban los
rankings por ruido estadístico, no por calidad real.

### `squad_builder.py`
El módulo más complejo — arma dos recomendaciones de equipo distintas,
ambas modeladas como problemas de **programación lineal entera (ILP)**
resueltos con PuLP/CBC en vez de heurísticas simples, porque las reglas de
FPL se cruzan entre sí (presupuesto, composición de posiciones, límite de
club, enfrentamientos internos) de una forma que un ranking no resuelve
bien — un solver garantiza la solución matemáticamente óptima.

**Equipo Base** (`select_squad` + `select_starting_xi`, dos pasos):
1. ILP: elige los 15 jugadores que maximizan el xP total sujeto a
   presupuesto (£100m), exactamente 2 GKP/5 DEF/5 MID/3 FWD, máximo 3 por
   club, y sin enfrentamientos internos.
2. Fuerza bruta: de esos 15, prueba las ~13 formaciones válidas
   (3-5 DEF, 2-5 MID, 1-3 FWD) y elige la de mayor xP para los 11
   titulares. No hace falta un solver acá — el espacio de búsqueda ya es
   chico porque el plantel está fijo.

Pensado como base de temporada: reparte el presupuesto entre los 15, así
el banco es jugable si hace falta usarlo en varias fechas.

**Equipo Wildcard** (`select_wildcard_squad`, un solo ILP): asume que se
puede rearmar el plantel entero cada fecha, así que el banco no necesita
ser jugable. Un único problema de optimización decide plantel y titulares
a la vez (variables `in_squad` e `in_xi`, con la restricción
`in_xi ≤ in_squad`): maximizar el xP del XI es la prioridad absoluta
(peso `WILDCARD_XI_WEIGHT = 100 000` en la función objetivo), y minimizar
el costo total del plantel es el criterio de desempate — empuja el gasto
del banco al mínimo posible sin sacrificar nunca un punto de xP en el XI.

*(Separar esto en dos pasos, como el equipo Base, no funciona acá: para
saber cuánto presupuesto "reservar" para el banco habría que saber antes
qué banco hace falta — es circular. Resolviendo todo en un solo ILP, el
solver lo maneja sin ese problema.)*

**Restricción anti-enfrentamiento** (`_add_no_internal_clash_constraints`,
compartida por ambos equipos): impide elegir a la vez al defensor/arquero
de un equipo y a un mediocampista/delantero del rival que enfrenta ese
mismo gameweek. Si ese rival anota, le rompe el clean sheet al propio
defensor — dos jugadores del mismo plantel "compitiendo" entre sí. En el
Wildcard, esta restricción solo aplica entre titulares (un suplente que
no juega no le puede romper el clean sheet a nadie).

### `photos.py`
Cada jugador tiene una foto potencial en el CDN de la Premier League,
armada a partir de su campo `code` (no viene como URL directa en la API).
Dos problemas encontrados y resueltos acá:

- **~35% de los jugadores no tienen foto todavía** (fichajes recién
  llegados sin sesión oficial). No hay forma de saberlo de antemano por
  campo alguno de la API — hay que consultar cada URL. `resolve_photo_urls`
  lo hace en paralelo (`ThreadPoolExecutor`, 50 workers) contra una única
  `requests.Session` con pool de conexiones — sin esto, cada una de las
  ~590 verificaciones abre su propia conexión TCP/TLS y el proceso tarda
  ~84 segundos; con sesión compartida baja a ~7-8 segundos. Las que fallan
  se reemplazan por la silueta genérica oficial de FPL
  (`Photo-Missing.png`, la misma que usa fantasy.premierleague.com).
- **Tamaño de imagen**: Tableau Public rechaza imágenes de más de 128KB.
  El tamaño 250x250 del CDN pesa ~300-350KB y 110x140 pesa ~95-145KB
  (a veces supera el límite, depende del detalle de la foto puntual, no
  solo de la resolución). Se usa 40x40 (~15-25KB) por margen de sobra.

Esta verificación solo corre en `export_for_tableau.py` — el resto del
pipeline (análisis, armado de equipo) no depende de esta llamada de red
extra.

## Decisiones de diseño y limitaciones conocidas

- **xP es una heurística de aprendizaje, no un modelo predictivo serio.**
  FPL expone su propia estimación en `ep_next` — comparar contra eso es
  una mejora pendiente.
- **El equipo Base optimiza el xP total de los 15**, no específicamente
  el de los 11 mejores — en la práctica puede "gastar de más" en el 5to
  defensor o 2do arquero en vez de un banco más barato.
- **El horizonte de fixtures es de un solo gameweek** (el próximo). Se
  evaluó extenderlo a 2 gameweeks pero se descartó: el objetivo real era
  que el plantel no se auto-sabotee (enfrentamientos internos), no
  optimizar matemáticamente a 2 fechas — la sostenibilidad de temporada
  se resuelve con las transferencias normales de FPL, no recalculando
  todo desde cero cada semana.
- **`MIN_MINUTES_FOR_RANKING`** filtra por muestra chica, no por
  potencial — un debutante prometedor sin minutos la temporada pasada
  queda fuera de las recomendaciones aunque sea una gran opción real.

## Cómo correr todo

```bash
pip install -r requirements.txt

python scripts/run_fetch.py           # descarga datos frescos de la API
python scripts/run_analysis.py        # xP, diferenciales, capitanía
python scripts/run_squad.py           # equipo Base
python scripts/run_wildcard_squad.py  # equipo Wildcard
python scripts/export_for_tableau.py  # CSVs finales con fotos resueltas
```

Salidas relevantes en `data/processed/`: `players_scored.csv` (todo el
pool de jugadores con xP) y `squad_recommendations.csv` (ambos equipos,
formato tidy, listo para Tableau).
