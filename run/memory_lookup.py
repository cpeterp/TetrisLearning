""" Memory addresses and corresponding values. Pulled primarly from 
https://github.com/h3nnn4n/Reverse-Engineering-the-GameBoy-Tetris/ """

# Tetris uses a single byte to store the shape and rotation of active and
# preview pieces
ACTIVE_SHAPE_ADDR = 0xC203
PREVIEW_SHAPE_ADDR = 0xC213
NEXT_PREVIEW_SHAPE_ADDR = 0xFFAE
# Shapes values are coded as 4x[lookup key] + [rotation value]
# E.g. value of 0x13 (19 dec) corresponds to 19//4 = 4 -> "Z", 19%4 -> 3 rot.
SHAPE_LOOKUP = {
    0: "L",
    1: "J",
    2: "I",
    3: "O",
    4: "Z",
    5: "S",
    6: "T",
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
