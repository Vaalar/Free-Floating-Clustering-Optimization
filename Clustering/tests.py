import PySimpleGUI as sg

layout = [
    [sg.Text('Progress Bar Example')],
    [sg.ProgressBar(100, orientation='h', size=(20, 20), key='-PROGRESS-')],
    [sg.Text('', size=(15, 1), key='-PROGRESS_TEXT-')],
    [sg.Button('Start'), sg.Button('Exit')],
]

window = sg.Window('Progress Bar Example', layout, finalize=True)

while True:
    event, values = window.read()

    if event in (sg.WINDOW_CLOSED, 'Exit'):
        break
    elif event == 'Start':
        for i in range(101):
            window['-PROGRESS-'].update(i)
            window['-PROGRESS_TEXT-'].update(f'Progress: {i}%')
            window.Refresh()  # Force a GUI refresh
            sg.popup_animated(sg.DEFAULT_BASE64_LOADING_GIF, location=(300, 150))
        sg.popup_animated(None)

window.close()
