========================
Minimal "viable" product
========================

**Minimal simulation**

- Modeled entities
    - Map
        - 2D grid
        - Initialized deterministically
        - Terrain types
            - Forest
            - High mountains
            - Desert
            - Sea
            - Grassland
        - Elevation?
        - Villages are distributed sparsely across the map
        - The probability of a square having a village depends on its terrain
          type
    - Villages
        - Population
            - TODO
        - Resource stores
            - TODO
        - Resource types
            - Wheat
            - Wood
            - Stone
            - Bronze
            - Gold
        - Farming
            - Model crop cultivation from sowing to harvest
            - TODO
        - Livestock
            - Model "herd" size
            - Should have seasonal birth cycle
            - Slaughtering converts livestock to meat resource
        - Forestry
            - TODO
        - Mining
            - TODO
        - Trade
            - TODO
        - Taxation
            - TODO
    - Weather
        - Should reflect unified weather patterns for the local area
        - Weather should be highly correlated both spatially and temporally
        - Should have defined seasonal variation
        - Weather types
            - Temperature
            - Rainfall

**Minimal UI**

The game will be played by installing the python library and running a command
in the terminal. Once in the game there will be commands that a player can
execute.

- Commands
    - Advance game
        - Advance `n` days (by default 1)
    - Perform action
        - TODO
    - Get information
        - Information about player's village
            - TODO
        - Current stores of all resources
            - TODO
        - Current population
            - TODO
        - Weather
            - TODO
    - Observe map
        - Should there be different map modes initially?
        - Later we can introduce fog-of-war
    - Exit