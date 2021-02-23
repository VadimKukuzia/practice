import PySimpleGUI as sg
import cv2

"""

"""


def main():
    sg.theme('BlueMono')

    # define the window layout
    layout = [
        [sg.Image(filename='', key='-IMAGE-')],
        [sg.Radio('None', 'Radio', True, size=(10, 1))],
        [sg.Radio('Threshold', 'Radio', size=(15, 1), key='-THRESH-'),
         sg.Slider((0, 255), 128, 1, orientation='h', size=(40, 15), key='-THRESH SLIDER-')],
        [sg.Radio('Gaussian Blurring', 'Radio', size=(15, 1), key='-BLUR-'),
         sg.Slider((1, 11), 1, 1, orientation='h', size=(20, 15), key='-BLUR SLIDER-')],
        [sg.Radio('Edge Detection', 'Radio', size=(15, 1), key='-EDGE-')],
        [sg.Radio('Bitwise Operations', 'Radio', size=(15, 1), key='-BITWISE-'),
         sg.Combo(['AND', 'OR', 'NOT', 'XOR'], size=(10, 1), key='-BITWISE_CHOICE-', default_value='AND')],
    ]

    # create the window and show it without the plot
    window = sg.Window('Process images of a video using OpenCV Python', layout)

    cap = cv2.VideoCapture('example.mp4')

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        ret, frame = cap.read()

        if values['-THRESH-']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.adaptiveThreshold(frame, values['-THRESH SLIDER-'], cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
        elif values['-BLUR-']:
            frame = cv2.GaussianBlur(frame, (21, 21), values['-BLUR SLIDER-'])

        elif values['-EDGE-']:
            frame = cv2.Canny(frame, 100, 200)

        elif values['-BITWISE-']:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            if values['-BITWISE_CHOICE-'] == 'AND':
                frame = cv2.bitwise_and(frame, frame, mask=mask)

            if values['-BITWISE_CHOICE-'] == 'OR':
                frame = cv2.bitwise_or(frame, frame, mask=mask)

            if values['-BITWISE_CHOICE-'] == 'NOT':
                frame = cv2.bitwise_not(frame, frame, mask=mask)

            if values['-BITWISE_CHOICE-'] == 'XOR':
                frame = cv2.bitwise_xor(frame, frame, mask=mask)

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)

    window.close()


main()
