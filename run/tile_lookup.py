""" Tile IDs exposed during main game and new IDs passed to agent observations. 
Tilemap IDs are returned by PyBoy.botsupport.tilemap_background(). """

# Define lookups for Tetris tilemap
TILEMAP_BLANK_TILE = 47
TILEMAP_SHAPE_TILES = [n for n in range(128, 143 + 1)]

# Define output values for observations
OBS_BLANK_TILE = 0
OBS_WALL_TILE = 126
OBS_SHAPE_TILE = 200
OBS_SPRITE_TILE = 255
OBS_SPACE_RANGE = (0, 255)
