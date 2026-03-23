"""Static PGA Tour constants used by ingestion and feature prep."""

from __future__ import annotations

from dataclasses import dataclass

PGA_TOUR_BASE_URL = "https://www.pgatour.com"
PGA_STATS_PATH = "/stats"
PGA_STAT_DETAIL_PATH = "/stats/detail/{stat_id}"
PGA_SCHEDULE_PATH = "/schedule"
PGA_SCHEDULE_SEASON_PATH = "/schedule/{season}"
PGA_TOURNAMENT_LEADERBOARD_PATH = "/tournaments/{season}/{slug}/{tournament_id}/leaderboard"
PGA_PLAYER_PROFILE_PATH = "/player/{player_id}"
DEFAULT_TOUR_CODE = "R"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 SeekingBetaAI/1.0"
)


@dataclass(frozen=True)
class TrackedPGAStat:
    """A curated PGA Tour stat to collect for model features."""

    stat_id: str
    slug: str
    title: str


DEFAULT_TRACKED_STATS: tuple[TrackedPGAStat, ...] = (
    TrackedPGAStat("02675", "sg_total", "SG: Total"),
    TrackedPGAStat("02674", "sg_tee_to_green", "SG: Tee-to-Green"),
    TrackedPGAStat("02567", "sg_off_the_tee", "SG: Off-the-Tee"),
    TrackedPGAStat("02568", "sg_approach", "SG: Approach the Green"),
    TrackedPGAStat("02569", "sg_around_green", "SG: Around-the-Green"),
    TrackedPGAStat("02564", "sg_putting", "SG: Putting"),
    TrackedPGAStat("120", "scoring_average", "Scoring Average"),
    TrackedPGAStat("101", "driving_distance", "Driving Distance"),
    TrackedPGAStat("102", "driving_accuracy_pct", "Driving Accuracy Percentage"),
    TrackedPGAStat("02438", "good_drive_pct", "Good Drive Percentage"),
    TrackedPGAStat("103", "gir_pct", "Greens in Regulation Percentage"),
    TrackedPGAStat("331", "proximity_to_hole", "Proximity to Hole"),
    TrackedPGAStat("130", "scrambling_pct", "Scrambling"),
    TrackedPGAStat("111", "sand_save_pct", "Sand Save Percentage"),
    TrackedPGAStat("119", "putts_per_round", "Putts Per Round"),
    TrackedPGAStat("413", "one_putt_pct", "One-Putt Percentage"),
    TrackedPGAStat("426", "three_putt_avoidance", "3-Putt Avoidance"),
    TrackedPGAStat("115", "birdie_or_better_conversion_pct", "Birdie or Better Conversion Percentage"),
    TrackedPGAStat("138", "top_10_finishes", "Top 10 Finishes"),
    TrackedPGAStat("300", "victory_leaders", "Victory Leaders"),
    TrackedPGAStat("186", "owgr", "Official World Golf Ranking"),
    TrackedPGAStat("127", "all_around_ranking", "All-Around Ranking"),
    TrackedPGAStat("171", "par_3_performance", "Par 3 Performance"),
    TrackedPGAStat("172", "par_4_performance", "Par 4 Performance"),
    TrackedPGAStat("173", "par_5_performance", "Par 5 Performance"),
    TrackedPGAStat("142", "par_3_scoring_average", "Par 3 Scoring Average"),
    TrackedPGAStat("143", "par_4_scoring_average", "Par 4 Scoring Average"),
    TrackedPGAStat("144", "par_5_scoring_average", "Par 5 Scoring Average"),
    TrackedPGAStat("112", "par_3_birdie_or_better", "Par 3 Birdie or Better Leaders"),
    TrackedPGAStat("113", "par_4_birdie_or_better", "Par 4 Birdie or Better Leaders"),
    TrackedPGAStat("114", "par_5_birdie_or_better", "Par 5 Birdie or Better Leaders"),
    TrackedPGAStat("156", "birdie_average", "Birdie Average"),
    TrackedPGAStat("352", "birdie_or_better_pct", "Birdie or Better Percentage"),
    TrackedPGAStat("160", "bounce_back", "Bounce Back"),
    TrackedPGAStat("104", "putting_average", "Putting Average"),
    TrackedPGAStat("402", "overall_putting_average", "Overall Putting Average"),
    TrackedPGAStat("398", "one_putts_per_round", "1-Putts per Round"),
    TrackedPGAStat("484", "putting_inside_10ft", "Putting - Inside 10'"),
    TrackedPGAStat("403", "putting_inside_5ft", "Putting from Inside 5'"),
    TrackedPGAStat("404", "putting_5_to_10ft", "Putting from 5-10'"),
    TrackedPGAStat("405", "putting_10_to_15ft", "Putting from 10-15'"),
    TrackedPGAStat("406", "putting_15_to_20ft", "Putting from 15-20'"),
    TrackedPGAStat("407", "putting_20_to_25ft", "Putting from 20-25'"),
    TrackedPGAStat("431", "fairway_proximity", "Fairway Proximity"),
    TrackedPGAStat("374", "arg_proximity", "Proximity to Hole (ARG)"),
    TrackedPGAStat("381", "arg_proximity_10_20", "Proximity to Hole from 10-20 yards"),
    TrackedPGAStat("380", "arg_proximity_20_30", "Proximity to Hole from 20-30 yards"),
    TrackedPGAStat("382", "arg_proximity_under_10", "Proximity to Hole from < 10 yards"),
    TrackedPGAStat("375", "arg_proximity_sand", "Proximity to Hole from Sand"),
    TrackedPGAStat("376", "arg_proximity_rough", "Proximity to Hole from Rough"),
    TrackedPGAStat("363", "scrambling_from_rough", "Scrambling from the Rough"),
    TrackedPGAStat("362", "scrambling_from_sand", "Scrambling from the Sand"),
    TrackedPGAStat("366", "scrambling_over_30", "Scrambling from > 30 yards"),
    TrackedPGAStat("481", "scrambling_avg_distance", "Scrambling Average Distance to Hole"),
    TrackedPGAStat("148", "round_1_scoring_average", "Round 1 Scoring Average"),
    TrackedPGAStat("149", "round_2_scoring_average", "Round 2 Scoring Average"),
    TrackedPGAStat("117", "round_3_scoring_average", "Round 3 Scoring Average"),
    TrackedPGAStat("285", "round_4_scoring_average", "Round 4 Scoring Average"),
    TrackedPGAStat("339", "approach_125_150_fairway", "Approaches from 125-150 yards"),
    TrackedPGAStat("338", "approach_150_175_fairway", "Approaches from 150-175 yards"),
    TrackedPGAStat("337", "approach_175_200_fairway", "Approaches from 175-200 yards"),
    TrackedPGAStat("360", "birdie_better_125_150", "Birdie or Better Percentage - 125-150 yards"),
    TrackedPGAStat("359", "birdie_better_150_175", "Birdie or Better Percentage - 150-175 yards"),
    TrackedPGAStat("358", "birdie_better_175_200", "Birdie or Better Percentage - 175-200 yards"),
    TrackedPGAStat("357", "birdie_better_200_plus", "Birdie or Better Percentage - 200+ yards"),
    TrackedPGAStat("361", "birdie_better_under_125", "Birdie or Better Percentage - < 125 yards"),
    TrackedPGAStat("106", "total_eagles", "Total Eagles"),
    TrackedPGAStat("107", "total_birdies", "Total Birdies"),
    TrackedPGAStat("447", "par_4_eagle_leaders", "Par 4 Eagle Leaders"),
    TrackedPGAStat("448", "par_5_eagle_leaders", "Par 5 Eagle Leaders"),
    TrackedPGAStat("317", "driving_distance_all_drives", "Driving Distance - All Drives"),
)
