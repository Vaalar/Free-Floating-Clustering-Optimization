import PySimpleGUI as sg
import threading

# Function that represents the main application logic
def main_application_logic():
    # Your main application logic here
    while True:
        sg.popup("Main Application Logic is running!")

# Define the layout of the GUI
layout = [
    [sg.Text("Main Application Logic in a Thread")],
    [sg.Button('Start')],
    [sg.Exit()]
]

window = sg.Window('Threaded GUI', layout)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    if event == 'Start':
        # Create a thread for the main application logic
        t = threading.Thread(target=main_application_logic)
        t.start()

window.close()
