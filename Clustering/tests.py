import PySimpleGUI as sg

layout = [
    [sg.Text('Elemento que se expandirá'), sg.InputText(key='-EXPAND-', size=(20, 1), expand_x=True)],
    [sg.Button('Enviar'), sg.Button('Cancelar')]
]

window = sg.Window('Ejemplo de expansión', layout, resizable=True, finalize=True)

# Definir la expansión máxima del elemento en dirección X e Y

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Cancelar':
        break

window.close()
