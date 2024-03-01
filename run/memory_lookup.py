""" Memory addresses and corresponding values. Pulled primarly from 
https://github.com/h3nnn4n/Reverse-Engineering-the-GameBoy-Tetris/ """

# Tetris uses a single byte to store the shape and rotation of active and
# preview pieces
ACTIVE_SHAPE_ADDR = 0xC203
PREVIEW_SHAPE_ADDR = 0xC213
NEXT_PREVIEW_SHAPE_ADDR = 0xFFAE
SHAPE_LOOKUP = {
    0x00: {"shape": "L", "rotation": 0},
    0x01: {"shape": "L", "rotation": 1},
    0x02: {"shape": "L", "rotation": 2},
    0x03: {"shape": "L", "rotation": 3},
    0x04: {"shape": "J", "rotation": 0},
    0x05: {"shape": "J", "rotation": 1},
    0x06: {"shape": "J", "rotation": 2},
    0x07: {"shape": "J", "rotation": 3},
    0x08: {"shape": "I", "rotation": 0},
    0x09: {"shape": "I", "rotation": 1},
    0x0A: {"shape": "I", "rotation": 2},
    0x0B: {"shape": "I", "rotation": 3},
    0x0C: {"shape": "O", "rotation": 0},
    0x0D: {"shape": "O", "rotation": 1},
    0x0E: {"shape": "O", "rotation": 2},
    0x0F: {"shape": "O", "rotation": 3},
    0x10: {"shape": "Z", "rotation": 0},
    0x11: {"shape": "Z", "rotation": 1},
    0x12: {"shape": "Z", "rotation": 2},
    0x13: {"shape": "Z", "rotation": 3},
    0x14: {"shape": "S", "rotation": 0},
    0x15: {"shape": "S", "rotation": 1},
    0x16: {"shape": "S", "rotation": 2},
    0x17: {"shape": "S", "rotation": 3},
    0x18: {"shape": "T", "rotation": 0},
    0x19: {"shape": "T", "rotation": 1},
    0x1A: {"shape": "T", "rotation": 2},
    0x1B: {"shape": "T", "rotation": 3},
}

# Tetris records its score as a 3 byte little endian BCD starting at 0xC0A0.
SCORE_ADDR_0 = 0xC0A0
SCORE_ADDR_1 = 0xC0A1
SCORE_ADDR_2 = 0xC0A2

# Tetris also records lines cleared as a 3 byte little endian BCD starting at 0xFF9E.
LINES_CLEARED_ADDR_0 = 0xFF9E
LINES_CLEARED_ADDR_1 = 0xFF9F
LINES_CLEARED_ADDR_2 = 0xFFE7

# Screen states are sored in a single byte.
SCREEN_STATE_ADDR = 0xFFE1
GAMEOVER_SCREEN_STATE = 0x04
GAMEOVER_ANIMATION_SCREEN_STATE = 0x0D

# Game Level
GAME_LEVEL_ADDR = 0xFFA9

# This byte appears to indicate when a shape is processing - i.e. when a shape
# has dropped to the bottom level it changes value to 0x80, otherwise it is 0x00
ACTIVE_SHAPE_FLAG_ADDR = 0xC200
INACTIVE_SHAPE_FLAG = 0x80
ACTIVE_SHAPE_FLAG = 0x00

# Tetris has a timer that counts down frames until the shape drops a level. Each
# level has a different delay, and this value is stored next to it. When the
# timer equals the delay, we know the shape will drop on the next frame.
DROP_TIMER_ADDR = 0xFF99
DROP_DELEAY_ADDR = 0xFF9A

DROP_STATE_ADRR = 0xFF98
SHAPE_DROPPING = 0x00
SHAPE_DROPPED = 0x02  # i.e. hit bottom
# Maintains this state until the next shape falls (plus 2 frames)
NEXT_SHAPE_PENDING = 0x03
