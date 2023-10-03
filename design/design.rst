**Game plan**

Player plays as the leader of a village. Player enacts policies that affect the
village. Could be loosely modeled on the world of the Sumerians, Vedic Aryans,
or some other similar civilization.

Goal:

- Improve village economy, defenses, military capability, etc.
- Take control of neighboring villages and build a kingdom

Methodology:

- Game engine is vivarium
- Vivarium simulates villages as a whole(?)
- Could consider updating vivarium to support multiple state tables allowing
  weather to be modeled using a different simulation methodology
    - This could be done by defining a new `population` plugin rather than
      directly modifying vivarium

Game aspects:

- Models fundamental aspects of the world
- Policies/actions that can be enacted or taken that impact the world

Modeled entities:

- Weather/seasons/climate
    - Could affect
        - Crop yield
        - Harvest date(s)
        - Mortality
        - Trade?
        - Natural disasters
- Plagues/pestilences
- Population/mortality/fertility
- Crop yield and food stores
- Livestock
- Trade?
- Mining - need access to metal for weapons and agricultural tools (iron or
  bronze?)
- Timber
- Army morale?
- Population loyalty?

Policies/actions:

- Muster armies
- Disband armies
- Declare war
- Build walls
- Taxation
- Irrigation
- Build public works
- Build roads/bridges
- Mete out justice

Thoughts/Questions/Ideas:

- How to incorporate religion?
- Fishing/navy?
- How to model geography?
    - Natural features like rivers, mountains, coastlines, forests, deserts
    - Actual physical location - i.e. location within a grid(?)
- Should there be a map for the player? Maybe not, since actual leaders at time
  wouldn't have had good maps?
- Are there different classes that should be modeled?
- What should harvest quantities be dependent on?
    - Population?
    - Land under cultivation?
    - Livestock?
- Harvest should be collected over time, not all at once (and per crop)
- What should inter-village interactions look like?
- Should we have nomadic/mountain raiders?
- How to model interactions with other villages/polities?
    - Diplomacy
    - War
    - Opinions