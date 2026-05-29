"""Summer Olympics modeling package.

Olympics spans dozens of disciplines, most of them individual timed/judged
events that don't fit a head-to-head classifier. The tractable, established
product is country medal-count forecasting, which is what this package does:
aggregate historical Summer Games medal tables and project the next edition.

Surfaced as its own ``olympics`` sport key whose single board is a ranked
"field" of countries by projected medals (rendered like a golf/tennis field).
"""
