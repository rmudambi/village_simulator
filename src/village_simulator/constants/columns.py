class Columns:
    # Map columns
    X: str = "x"
    Y: str = "y"
    TERRAIN: str = "terrain"

    # Village columns
    IS_VILLAGE: str = "is_village"
    ARABLE_LAND: str = "arable_land"

    # Weather columns
    TEMPERATURE: str = "temperature"
    RAINFALL: str = "rainfall"

    # Demographics columns
    FEMALE_POPULATION_SIZE = "female_population_size"
    MALE_POPULATION_SIZE = "male_population_size"

    # Resource columns
    @staticmethod
    def get_resource_stores(resource: str) -> str:
        return f"{resource}_stores"

    # Wheat columns
    PROJECTED_WHEAT_HARVEST: str = "projected_wheat_harvest"

    # Wheat growth columns
    PREVIOUSLY_DRY = "previous_day_dry"
    CUMULATIVE_DRY_DAYS = "cumulative_dry_days"
    RAINFALL_MID_GROWTH = "rainfall_mid_growth"
    RAINFALL_LATE_GROWTH = "rainfall_late_growth"
