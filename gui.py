import time

import PySimpleGUI as sg
import find_patterns
import librosa
import sounddevice as sd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


sg.theme('DarkGreen')
button_size = (8, 1)


def thread_stop_rec(button, duration):
    time.sleep(duration)
    button.update(text='Record')


layout = [
    [sg.Text('Audio-pattern')],
    [sg.Input(key='Input1'), sg.FileBrowse(size=button_size),
     sg.Button("Record/Stop", size=(10, 1)), sg.Button("Play", size=button_size)],
    [sg.Text('Audio-stream')],
    [sg.Input(key='Input2'), sg.FileBrowse(size=button_size),
     sg.Button("Record/Stop", size=(10, 1)), sg.Button("Play", size=button_size)],
    [sg.Button("Run", size=button_size)],
    [sg.Canvas(size=(600, 300), pad=(6, 10), background_color='white', key='graph')]
]

window = sg.Window('Sound detection', layout)
window.Size = (700, 400)
window.Finalize()
sr = 22050
p_seconds, au_seconds = 1, 10
pattern, audio = None, None
graph = None
last_event = None
p_start, au_start = None, None
last1, last = None, None
fig, ax = plt.subplots(1, 1, figsize=(6, 3))

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break

    elif event == "Record/Stop":

        if p_start is None or time.time() - p_start > p_seconds:
            p_start = time.time()
            pattern = sd.rec(int(p_seconds * sr), samplerate=sr, channels=1).reshape(-1)
            # recording1 = True
            window['Input1'].update(value='')
            continue

        elif p_start is not None:
            sd.stop()
            pattern = pattern[:int((time.time() - p_start) * sr)]

    elif event == "Record/Stop1":

        if au_start is None or time.time() - au_start > au_seconds:
            au_start = time.time()
            audio = sd.rec(int(au_seconds * sr), samplerate=sr, channels=1).reshape(-1)
            last2 = "Record"
            window['Input2'].update(value='')
            continue

        elif p_start is not None:
            sd.stop()
            audio = audio[:int((time.time() - au_start) * sr)]

    elif event == "Play":

        if window['Input1'].Get() and window['Input1'].Get().endswith('.wav'):
            pattern, _ = librosa.load(window['Input1'].Get())

        sd.play(pattern, sr)
        status = sd.wait()

    elif event == "Play2":

        if window['Input2'].Get() and window['Input2'].Get().endswith('.wav'):
            audio, _ = librosa.load(window['Input2'].Get())

        sd.play(audio, sr)
        status = sd.wait()

    elif event == "Run":

        if graph is not None:
            graph.get_tk_widget().forget()

        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        fig.tight_layout(pad=2.0)

        if window['Input1'].Get():
            pattern = window['Input1'].Get()
        if window['Input2'].Get():
            audio = window['Input2'].Get()

        find_patterns.find_patterns(pattern, audio)
        graph = draw_figure(window['graph'].TKCanvas, fig)
        graph.draw()

window.close()
