"""Association football (soccer) modeling package.

Soccer is modeled as ONE sport whose ``tour`` carries the competition
(Premier League, La Liga, Serie A, Bundesliga, Ligue 1, UEFA Champions
League, FIFA World Cup), mirroring how golf/tennis use ``tour``.

Match outcomes are 1X2 (home win / draw / away win), produced by a
Dixon-Coles bivariate-Poisson goal model (see ``dixon_coles``).
"""
