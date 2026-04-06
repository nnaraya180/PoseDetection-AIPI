# lcd_i2c.py
# Shim matching lcd_i2c.LCD_I2C interface, backed by i2c_lcd.lcd
# Same local-shadow trick used for picamzero.py

from i2c_lcd import lcd as _lcd

class LCD_I2C:
    def __init__(self, addr, cols, rows):
        self._lcd  = _lcd(addr)
        self._cols = cols
        self.backlight = _Backlight(self._lcd)
        self.cursor    = _Cursor(self._lcd, cols)

    def write_text(self, text):
        # i2c_lcd overwrites from current cursor position
        self._lcd.lcd_display_string(text, self.cursor._row + 1)


class _Backlight:
    def __init__(self, lcd):
        self._lcd = lcd

    def on(self):
        self._lcd.backlight_on(True)

    def off(self):
        self._lcd.backlight_on(False)


class _Cursor:
    def __init__(self, lcd, cols):
        self._lcd  = lcd
        self._cols = cols
        self._row  = 0

    def setPos(self, row, col):
        self._row = row
