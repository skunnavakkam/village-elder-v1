from display import TextBufferEPD
import epaper
import time
epd = epaper.epaper('epd2in13_V4').EPD()


epd.init()


text_buffer = TextBufferEPD(epd)

text_buffer.write("Hello World")
time.sleep(2)
text_buffer.write(" my name is")
time.sleep(2)
text_buffer.write(" John Doe")
time.sleep(2)
text_buffer.write(" I am a software engineer")
time.sleep(2)
text_buffer.write(" I love to code")
time.sleep(2)