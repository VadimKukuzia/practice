import PySimpleGUI as sg
import os
sg.theme('BlueMono')  # Add a touch of color
# All the stuff inside your window.
layout = [[sg.Text('1. Choose image to crop', size=(35, 1))],
          [sg.Text('2. Draw a rectangle with the mouse', size=(35, 1),)],
          [sg.Text('3. Press "C" for cut the highlighted area,', size=(45, 1))],
          [sg.Text('"R" - to reset the image, "Q" for quit from app', size=(50, 1))],
          [sg.Input(key='-IMAGE-IN-'), sg.FileBrowse()],
          [sg.Image(filename='', key='-IMAGE-OUT-')],
          [sg.Button('Open file'), sg.Button('Cancel')]]

# Create the Window
window = sg.Window('Draw rectangular shapes and extract objects', layout)
# Event Loop to process "events" and get the "values" of the inputs

# now let's initialize the list of reference point


while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        break

    if values['-IMAGE-IN-']:
        src = values['-IMAGE-IN-']
        os.system(f'python program_to_extract_particular_object_from_image.py --image {src}')


window.close()
