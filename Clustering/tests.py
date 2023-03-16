import PySimpleGUI as sg
import time
import sys

def longFunc():
    for i in range(15):
        print(i)
        print(i, file=sys.__stdout__) # output to terminal is working realtime
        time.sleep(1)

layout = [
        [sg.RButton('Long Func', key='longFunc')],
        [sg.Output(size=(60, 20))],
        ]

window = sg.Window('realtime output').Layout(layout)

while True:
    button, values = window.Read()
    print(button, values)

    if button is None or button == 'Exit':
        break
    elif button == 'longFunc':
        longFunc()