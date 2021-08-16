import time
import PySimpleGUI as sg
import librosa
import sounddevice as sd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import find_patterns
from run import isfloat


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def thread_stop_rec(button, duration):
    time.sleep(duration)
    button.update(text='Record')


if __name__ == "__main__":
    sg.theme('DarkGreen')
    view_size = (8, 1)

    layout = [
        [sg.Text('Audio-pattern')],
        [sg.Input(key='Input1'), sg.FileBrowse(size=view_size),
         sg.Button("Record/Stop", size=(10, 1)), sg.Button("Play", size=view_size)],
        [sg.Text('Audio-stream')],
        [sg.Input(key='Input2'), sg.FileBrowse(size=view_size),
         sg.Button("Record/Stop", size=(10, 1)), sg.Button("Play", size=view_size)],
        [sg.Text("Parameters")],
        [
            sg.Text("Sample rate:", tooltip="Sample rate of the audio-pattern and the audio-stream"),
            sg.Input(default_text='22050', size=view_size, key='sr'),
            sg.Text("n_mfcc:", tooltip="Number of mel-frequency cepstral coefficients"),
            sg.Input(default_text='20', size=view_size, key='n_mfcc'),
            sg.Text("Threshold:",
                    tooltip="Minimal degree of similarity between the pattern and the audio-stream window"),
            sg.Input(default_text='0.8', size=view_size, key='threshold'),
            sg.Text("Cut parameter:", tooltip="Used for audio-pattern cutting"),
            sg.Input(default_text='0.96', size=view_size, key='q')
        ],
        [sg.Button("Run", size=view_size)],
        [sg.Canvas(size=(600, 300), pad=(6, 10), background_color='white', key='graph', )]
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

            sr = window['sr'].Get()
            sr = int(sr) if sr in ['22050', '44100'] else 22050
            n_mfcc = window['n_mfcc'].Get()
            n_mfcc = int(n_mfcc) if n_mfcc.isnumeric() and int(n_mfcc) > 0 else 20
            threshold = window['threshold'].Get()
            threshold = float(threshold) if isfloat(threshold) and 0 <= float(threshold) <= 1 else 0.8
            q = window['q'].Get()
            q = float(q) if isfloat(q) and 0 <= float(q) <= 1 else 0.96
            if graph is not None:
                graph.get_tk_widget().forget()

            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            fig.tight_layout(pad=2.0)

            if window['Input1'].Get():
                pattern = window['Input1'].Get()
            if window['Input2'].Get():
                audio = window['Input2'].Get()
            if pattern is not None and audio is not None:
                find_patterns.find_patterns(pattern, audio, sr=sr, n_mfcc=n_mfcc, threshold=threshold, q=q)
                graph = draw_figure(window['graph'].TKCanvas, fig)
                graph.draw()

    window.close()
