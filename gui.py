import PySimpleGUI as sg
import librosa


def predict(filename):
    return 9


labels = ['Air conditioner', 'Car horn', 'Children playing', 'Dog bark', 'Drilling',
          'Engine idling', 'Gun shot', 'Jackhammer', 'Siren', 'Street music']
label_layout = [[sg.Text(label, key="label-{}".format(i))] for i, label in enumerate(labels)]
layout = [
    [sg.Text('Choose filename')],
    [sg.Input(), sg.FileBrowse()],
    [sg.Button("Predict"), sg.Button("Play")],
    [sg.Graph(canvas_size=(400, 400), graph_bottom_left=(-105, -105), graph_top_right=(10, 10),
              background_color='white', key='graph'),
     sg.Frame(layout=label_layout, title="Predicted label")]
]

window = sg.Window('Sound detection', layout)  # [[sg.Text('Filename')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
pred_label = 0
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == "Predict":
        window["label-{}".format(pred_label)].update(background_color=None, text_color=None)
        if values['Browse'][-4:] == '.wav':
            pred_label = predict(values['Browse'])
            window["label-{}".format(pred_label)].update(background_color='white', text_color='black')

window.close()


