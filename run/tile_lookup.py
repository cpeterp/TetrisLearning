"""
Tile IDs exposed during main game and new IDs passed to agent observations. 
Tilemap IDs are returned by PyBoy.botsupport.tilemap_background().
"""

TILEMAP_BLANK_TILE = 47
TILEMAP_SHAPE_TILES = [n for n in range(128, 143 + 1)]

OBS_BLANK_TILE = 0
OBS_SHAPE_TILE = 1
OBS_WALL_TILE = 2
